"""
Array-based population state.

The simulation used to keep one Python `Primate` object per individual. Looping over tens of thousands of objects is what
makes large populations slow, so the numerical state now lives in NumPy arrays instead: ages[i], is_female[i],
params_id[i], union_id[i] ... where `i` identifies an individual. Everything that is a property of the *species* rather
than of the individual (puberty age, lifespan, mortality rates ...) lives once per species in a `ParameterTable`, and an
individual just stores which row it uses (`params_id`).

    Python controller (PrimatePopulationSimulator.py)
              |
              v
    PopulationState  +  ParameterTable          <- this file: the data
              |
              v
    population_ops.py                           <- vectorized simulation operations
              |
              v
    graphing.py                                 <- reporting / plots

`Primate` and `Union` objects still exist, but only at the edges: loading a scenario, tests, and inspecting results
(`PopulationState.to_primates`). They are not used inside the simulation cycle.
"""
import numpy as np

from PopulationObjects import Primate, Union, earth_year

# Diet strings -> small integers so that feeding can be done with array masks.
DIET_CODES = {"autotroph": 0, "carnivore": 1, "herbivore": 2, "ruminant": 3, "omnivore": 4, "everything": 5}
UNKNOWN_DIET = -1   # an unrecognised diet string: the original feeding loop never feeds such an individual


class NameTable:
    """Interns species-name strings as small integers (the per-individual `species_id`)."""

    def __init__(self):
        self.names = []
        self._id_of = {}

    def id(self, name):
        found = self._id_of.get(name)
        if found is None:
            found = len(self.names)
            self.names.append(name)
            self._id_of[name] = found
        return found

    def __getitem__(self, name_id):
        return self.names[name_id]


class ParameterTable:
    """
    Species-level parameters stored as columns: `table.lifespan_days[params_id]` gathers the lifespan of many
    individuals at once. Row r corresponds to one `SimulationParameters` object (`table.objects[r]`).

    Hybrid offspring get their own, blended `SimulationParameters` (SimulationParameters.from_parents). They are added
    as new rows; identical blends are shared, so the table stays small.
    """
    FLOAT_FIELDS = (
        "puberty_age_days", "menopause_age_days", "lifespan_days", "lifespan_years", "human_aging_factor",
        "coupling_rate", "interbirth_interval_days", "max_kids_per_primate", "chance_of_multiple_birth",
        "sterile_chance", "sex_ratio_at_birth", "contraception_abortion_use_rate", "infant_mortality_rate",
        "maternal_mortality_rate", "adult_mortality_rate", "calories_needed_per_primate", "genetic_diversity",
        "effective_per_cycle_fertility_rate", "fertility_rising_steepness", "fertility_falling_steepness",
        "fertility_rising_midpoint", "fertility_declining_midpoint",
    )
    BOOL_FIELDS = (
        "is_hermaphrodite", "is_sequential_species", "ages_backward", "is_dominant", "lopsided_sex_lifespan",
        "has_sequent_sex_transition", "respawn_as_male", "is_aquatic", "kid_fodder",
        "is_humanoid", "is_xenophilic", "flat_fertility",
    )
    # Derived columns that are not plain attributes of SimulationParameters.
    EXTRA_BOOL = ("seasonal_not_false", "seasonal_truthy")   # see _row_values
    EXTRA_INT = ("diet_code", "name_id")

    def __init__(self, names):
        self.names = names
        self.objects = []      # row -> SimulationParameters
        self._row_of = {}      # id(SimulationParameters) -> row
        self._row_of_key = {}  # field-values tuple -> row (shares identical hybrid blends)
        self._cols = {}
        self._capacity = 0
        self.size = 0

    def __getattr__(self, name):
        # table.lifespan_days -> the column (all rows). Only called for names not found normally.
        cols = self.__dict__.get("_cols", {})
        if name in cols:
            return cols[name][:self.__dict__.get("size", 0)]
        raise AttributeError(name)

    # ---- registration -------------------------------------------------------------------------------------------
    def row_for(self, params):
        """Row of this exact object, registering it if new (founders and scenarios)."""
        row = self._row_of.get(id(params))
        if row is None:
            row = self._add(params)
        return row

    def row_for_equivalent(self, params):
        """Row of an object with identical values, if one exists (used for freshly created hybrid blends)."""
        row = self._row_of_key.get(self._key(params))
        if row is None:
            row = self._add(params)
        return row

    def _key(self, params):
        values, names = self._row_values(params)
        return (params.species_name,) + values + names

    def _row_values(self, params):
        floats = tuple(float(getattr(params, f)) for f in self.FLOAT_FIELDS)
        bools = tuple(bool(getattr(params, f)) for f in self.BOOL_FIELDS)
        extra_bool = (params.seasonal_mater is not False, bool(params.seasonal_mater))
        diet = DIET_CODES.get(params.diet_key, UNKNOWN_DIET)
        return floats + bools + extra_bool, (diet, self.names.id(params.species_name))

    def _add(self, params):
        if self.size == self._capacity:
            self._grow()
        row = self.size
        values, (diet, name_id) = self._row_values(params)
        fields = self.FLOAT_FIELDS + self.BOOL_FIELDS + self.EXTRA_BOOL
        for field, value in zip(fields, values):
            self._cols[field][row] = value
        self._cols["diet_code"][row] = diet
        self._cols["name_id"][row] = name_id
        self.objects.append(params)
        self._row_of[id(params)] = row
        self._row_of_key[(params.species_name,) + values + (diet, name_id)] = row
        self.size += 1
        return row

    def _grow(self):
        new_capacity = max(16, self._capacity * 2)
        for field in self.FLOAT_FIELDS:
            self._resize(field, np.float64, new_capacity)
        for field in self.BOOL_FIELDS + self.EXTRA_BOOL:
            self._resize(field, np.bool_, new_capacity)
        for field in self.EXTRA_INT:
            self._resize(field, np.int32, new_capacity)
        self._capacity = new_capacity

    def _resize(self, field, dtype, capacity):
        new = np.zeros(capacity, dtype=dtype)
        if field in self._cols:
            new[:self.size] = self._cols[field][:self.size]
        self._cols[field] = new


class PopulationState:
    """
    One array per per-individual attribute. Index i is one individual; all arrays always have the same length.

        age_days[i]           float   age in days (ages_backward species count down)
        is_female[i]          bool    sex (True for every hermaphrodite)
        species_id[i]         int     the individual's own species-name string (see NameTable)
        params_id[i]          int     row in the ParameterTable holding its species-level rules
        is_fertile[i]         bool    fertility state (False = sterile)
        children[i]           int     number of healthy children
        next_breeding_day[i]  float   earliest day this individual may breed again
        union_id[i]           int     label of the union the individual belongs to, -1 = single
        alive[i]              bool    operations mark deaths here; compaction (take/compress) then drops them

    `species_id` and `params_id` are deliberately separate: founders are named by their profile key while newborns are
    named by the profile's species_name, and the simulation (like the original code) compares those names when it
    decides whether a pairing is a hybrid.

    Union labels are stable integers handed out by `new_union_ids`. Because a label belongs to the individual (not to a
    position in the arrays), removing dead individuals never requires renumbering unions.
    """
    COLUMNS = (
        ("age_days", np.float64), ("is_female", np.bool_), ("species_id", np.int32), ("params_id", np.int32),
        ("is_fertile", np.bool_), ("children", np.int32), ("next_breeding_day", np.float64),
        ("union_id", np.int64), ("alive", np.bool_),
    )
    NO_UNION = -1

    def __init__(self, arrays=None, union_counter=None, context=None):
        arrays = arrays or {}
        self.context = context   # (ParameterTable, NameTable): lets reporting turn ids back into species names
        for name, dtype in self.COLUMNS:
            setattr(self, name, arrays[name] if name in arrays else np.zeros(0, dtype=dtype))
        self._union_counter = union_counter if union_counter is not None else [0]   # shared by copies of this state

    def __len__(self):
        return len(self.age_days)

    def new_union_ids(self, count):
        first = self._union_counter[0]
        self._union_counter[0] += count
        return np.arange(first, first + count, dtype=np.int64)

    def take(self, index):
        """A new state containing only the individuals selected by a boolean mask or an index array."""
        return PopulationState({name: getattr(self, name)[index] for name, _ in self.COLUMNS}, self._union_counter, self.context)

    def append(self, other):
        """A new state: this population followed by `other` (newborns)."""
        if len(other) == 0:
            return self
        return PopulationState({name: np.concatenate((getattr(self, name), getattr(other, name))) for name, _ in self.COLUMNS},
                               self._union_counter, self.context)

    @classmethod
    def newborns(cls, like, count, age_days, is_female, species_id, params_id, is_fertile):
        """A state of `count` new individuals (no children, no union, may breed immediately)."""
        return cls({
            "age_days": np.asarray(age_days, dtype=np.float64), "is_female": np.asarray(is_female, dtype=np.bool_),
            "species_id": np.asarray(species_id, dtype=np.int32), "params_id": np.asarray(params_id, dtype=np.int32),
            "is_fertile": np.asarray(is_fertile, dtype=np.bool_), "children": np.zeros(count, dtype=np.int32),
            "next_breeding_day": np.zeros(count, dtype=np.float64), "union_id": np.full(count, cls.NO_UNION, dtype=np.int64),
            "alive": np.ones(count, dtype=np.bool_),
        }, like._union_counter, like.context)

    # ---- reporting helpers (graphing.py) ----------------------------------------------------------------------------
    def ages_in_years(self):
        return self.age_days / earth_year

    def species_counts(self):
        """{species name: count}, in order of first appearance (the order the old report listed them)."""
        table, names = self.context
        if len(self) == 0:
            return {}
        ids = table.name_id[self.params_id]
        unique, first = np.unique(ids, return_index=True)
        counts = np.bincount(ids)
        return {names[int(i)]: int(counts[i]) for i in unique[np.argsort(first)]}

    # ---- conversion to and from Primate objects (scenarios, tests, inspection; not used inside the cycle) -----------
    @classmethod
    def from_primates(cls, primates, table, names, union_counter=None):
        state = cls({
            "age_days": np.array([p.age_days for p in primates], dtype=np.float64),
            "is_female": np.array([p.is_female for p in primates], dtype=np.bool_),
            "species_id": np.array([names.id(p.species_name) for p in primates], dtype=np.int32),
            "params_id": np.array([table.row_for(p.params) for p in primates], dtype=np.int32),
            "is_fertile": np.array([p.is_fertile for p in primates], dtype=np.bool_),
            "children": np.array([p.number_of_healthy_children for p in primates], dtype=np.int32),
            "next_breeding_day": np.array([p.next_breeding_day for p in primates], dtype=np.float64),
            "union_id": np.full(len(primates), cls.NO_UNION, dtype=np.int64),
            "alive": np.ones(len(primates), dtype=np.bool_),
        }, union_counter, (table, names))
        label_of = {}
        for i, p in enumerate(primates):
            if p.union is not None:
                if id(p.union) not in label_of:
                    label_of[id(p.union)] = int(state.new_union_ids(1)[0])
                state.union_id[i] = label_of[id(p.union)]
        return state

    def to_primates(self, table, names):
        """Materialise Primate (and Union) objects. Slow for large populations; meant for tests and inspection."""
        people = []
        for i in range(len(self)):
            p = Primate(names[int(self.species_id[i])], bool(self.is_female[i]), float(self.age_days[i]),
                        bool(self.is_fertile[i]), table.objects[int(self.params_id[i])])
            p.number_of_healthy_children = int(self.children[i])
            p.next_breeding_day = float(self.next_breeding_day[i])
            people.append(p)
        unions = {}
        for i, label in enumerate(self.union_id.tolist()):
            if label == self.NO_UNION:
                continue
            union = unions.get(label)
            if union is None:
                union = unions[label] = Union("monogamy", 2)
            union.max_size = max(union.max_size, len(union.members) + 1)
            union.add_member(people[i])
        return people
