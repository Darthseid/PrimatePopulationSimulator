"""
Vectorized simulation operations on a PopulationState.

Each function is one phase of a simulation cycle, in the order run_simulation() calls them:

    age_and_old_age_deaths  ->  count_by_group  ->  couple_singles  ->  reproduce  ->  final_deaths
        ->  feed  ->  clean_unions                                       (resolve_warfare runs before aging)

They take the PopulationState / ParameterTable plus an explicit numpy random Generator, and they never loop over
individuals in Python, except where the rules are inherently sequential and only a few individuals are involved
(hybrid parameter blends, warfare, the water pool). The rules themselves (formulas, order of events, which random
draws decide what) are those of the earlier object-based implementation; see the notes on each function for the places
where vectorising forces a documented difference.
"""
import math

import numpy as np

from PopulationObjects import earth_year as E

GOMPERTZ_A = 0.0001   # Old-age mortality: hazard per year = A * e^(B * biological age)
GOMPERTZ_B = 0.082
SEQUENT_TRANSITION_YEARS = 12783 / E   # Age at which sequential hermaphrodites switch sex
RESPAWN_AGE_DAYS = 4748                # "Doubles" come back as 13-year-old males
SEQUENT_RESET_AGE_DAYS = 5479          # age a sequential hermaphrodite has after switching sex (15 years)
DOUBLES_NAME = "Doubles"


# small shared helpers

def age_in_days_roundtrip(age_days):
    """`age_years * earth_year` as the original code computed it (same float round trip)."""
    return (age_days / E) * E


def caloric_need(state, table):
    """Daily resource need per individual (vector form of Primate.get_caloric_need)."""
    p = state.params_id
    need = table.calories_needed_per_primate[p]
    need = np.where(state.is_female, need * 0.9, need)
    biological_age_days = age_in_days_roundtrip(state.age_days)
    need = np.where(biological_age_days < table.puberty_age_days[p], need * 0.5, need)
    need = np.where(biological_age_days > table.lifespan_days[p], need * 0.75, need)
    return need


def count_adult_deaths(state, died, table):
    """(female_had_child, female_childless, male_had_child, male_childless) among the adults flagged in `died`."""
    adult = died & (age_in_days_roundtrip(state.age_days) >= table.puberty_age_days[state.params_id])
    had_child = state.children > 0
    female = state.is_female
    return (int(np.count_nonzero(adult & female & had_child)), int(np.count_nonzero(adult & female & ~had_child)),
            int(np.count_nonzero(adult & ~female & had_child)), int(np.count_nonzero(adult & ~female & ~had_child)))


def respawned_males(like, count, params_rows, species_ids, table, rng):
    """New 13-year-old males ("Doubles" come back as males when a female dies); fertile unless sterile by chance."""
    fertile = rng.random(count) > table.sterile_chance[params_rows]
    return like.newborns(like, count, np.full(count, RESPAWN_AGE_DAYS), np.zeros(count, dtype=bool), species_ids,
                         params_rows, fertile)

# 1. aging and old-age death

class AgingResult:
    def __init__(self, died, old_age_deaths, childless_tally, respawned):
        self.died = died                      # bool mask over the population that was aged
        self.old_age_deaths = old_age_deaths
        self.childless_tally = childless_tally  # see count_adult_deaths
        self.respawned = respawned            # PopulationState of respawned males (possibly empty)


def age_and_old_age_deaths(state, table, names, rng, cycle_days, cycle_length_years):
    """
    Ages everyone by one cycle (backwards for species that age backwards), applies sequential-hermaphrodite sex
    changes, then decides old-age death. Modifies `state.age_days`/`is_female`/`alive` in place.

    Gompertz mortality: hazard per year = A * exp(B * biological age), converted to a per-cycle probability.
    """
    p = state.params_id
    n = len(state)
    backward = table.ages_backward[p]
    age_days = state.age_days + np.where(backward, -float(cycle_days), float(cycle_days))
    age_years = age_days / E

    female_before = state.is_female
    widow_multiplier = np.where(~female_before & table.lopsided_sex_lifespan[p], 3, 1)   # uses the sex before any change
    changes_sex = table.has_sequent_sex_transition[p] & ~female_before & (age_years > SEQUENT_TRANSITION_YEARS)
    is_female = female_before | changes_sex
    age_days = np.where(changes_sex, float(SEQUENT_RESET_AGE_DAYS), age_days)
    age_years = age_days / E

    biological_age = (age_years * widow_multiplier) * table.human_aging_factor[p]
    biological_age = np.where(is_female, biological_age - 4.5, biological_age)
    hazard_per_year = GOMPERTZ_A * np.exp(GOMPERTZ_B * biological_age)
    mortality_per_cycle = 1.0 - np.exp(-hazard_per_year * cycle_length_years)
    mortality_per_cycle = np.where(table.lifespan_years[p] <= 0, 0.01, mortality_per_cycle)
    died = np.where(backward, age_days <= 0, rng.random(n) < mortality_per_cycle)   # backward-agers die at age 0

    state.age_days = age_days
    state.is_female = is_female
    state.alive = state.alive & ~died

    tally = count_adult_deaths(state, died, table)
    revive = died & is_female & table.respawn_as_male[p]
    respawned = respawned_males(state, int(np.count_nonzero(revive)), p[revive],
                                np.full(int(np.count_nonzero(revive)), names.id(DOUBLES_NAME)), table, rng)
    return AgingResult(died, int(np.count_nonzero(died)), tally, respawned)

# 2. group counts (used for the breeding-population estimate)

class GroupCounts:
    pass


def count_by_group(state, table, age_years):
    c = GroupCounts()
    p = state.params_id
    age_d = age_years * E
    fertile_age_male = state.is_fertile & (age_d >= table.puberty_age_days[p])
    fertile_age_female = state.is_fertile & (table.puberty_age_days[p] <= age_d) & (age_d < table.menopause_age_days[p])
    c.female = int(np.count_nonzero(state.is_female))
    c.male = len(state) - c.female
    c.fertile_female = int(np.count_nonzero(state.is_female & fertile_age_female))
    c.fertile_male = int(np.count_nonzero(~state.is_female & fertile_age_male))
    c.non_aquatic = int(np.count_nonzero(~table.is_aquatic[p]))
    c.seasonal = int(np.count_nonzero(table.seasonal_not_false[p]))   # complement of the `is False` off-season filter
    return c


def fertile_females_by_reference_rules(state, table, age_years, puberty_days, menopause_days):
    """Hermaphroditic species: females counted against one reference species' fertile ages (see controller)."""
    age_d = age_years * E
    return int(np.count_nonzero(state.is_fertile & (puberty_days <= age_d) & (age_d < menopause_days)))

# 3. unions

class UnionStats:
    """Per-union size and sex composition, derived from the individuals' union_id labels."""

    def __init__(self, state, table):
        self.coupled = np.flatnonzero(state.union_id >= 0)       # population indices of everyone in a union
        labels = state.union_id[self.coupled]
        self.labels, self.inverse = np.unique(labels, return_inverse=True)
        self.inverse = self.inverse.reshape(-1)
        count = len(self.labels)
        herm = table.is_hermaphrodite[state.params_id[self.coupled]]
        female_like = state.is_female[self.coupled] | herm        # Union.has_females()
        male_like = ~state.is_female[self.coupled] | herm         # Union.has_males()
        self.size = np.bincount(self.inverse, minlength=count)
        self.has_females = np.bincount(self.inverse, weights=female_like, minlength=count) > 0
        self.has_males = np.bincount(self.inverse, weights=male_like, minlength=count) > 0

    @property
    def dissolved(self):
        """Union.is_dissolved(): a union lacking a female or a male (a hermaphrodite counts as both)."""
        return ~(self.has_females & self.has_males)

    def per_individual(self, per_union, n, default=False):
        out = np.full(n, default, dtype=per_union.dtype)
        out[self.coupled] = per_union[self.inverse]
        return out

    def partners_of_pairs(self, n):
        """partner[i] = the other member of i's union if that union has exactly two members, else -1."""
        partner = np.full(n, -1, dtype=np.int64)
        position = np.arange(len(self.coupled))
        # In a two-member union the partner's position is (sum of both members' positions) - (my own position).
        position_sum = np.bincount(self.inverse, weights=position, minlength=len(self.labels))
        other = (position_sum[self.inverse] - position).astype(np.int64)
        in_pair = np.flatnonzero(self.size[self.inverse] == 2)
        partner[self.coupled[in_pair]] = self.coupled[other[in_pair]]
        return partner


def clean_unions(state, table):
    """End-of-cycle cleanup: every member of a dissolved union (e.g. one whose partner died) becomes single."""
    stats = UnionStats(state, table)
    if len(stats.coupled):
        state.union_id[stats.coupled[stats.dissolved[stats.inverse]]] = state.NO_UNION


def uncouple(state, mask):
    """A member leaves its union. Its former partner keeps the label (a one-person union) until clean_unions."""
    state.union_id[mask] = state.NO_UNION


def _distinct_samples(rng, rows, population, k):
    """`rows` independent random samples of k distinct positions out of range(population), in random order."""
    if k >= population or population <= 256:
        return np.argsort(rng.random((rows, population)), axis=1)[:, :k]
    samples = rng.integers(0, population, size=(rows, k))
    while True:   # redraw any row that picked the same position twice (rare once the population is large)
        ordered = np.sort(samples, axis=1)
        repeated = np.flatnonzero((ordered[:, 1:] == ordered[:, :-1]).any(axis=1))
        if repeated.size == 0:
            return samples
        samples[repeated] = rng.integers(0, population, size=(repeated.size, k))


MARRIAGE_MAX_SIZE = {"monogamy": 2, "polygyny": 5, "polyandry": 5, "polygamy": 9}


def _species_compatible(state, table, left, right):
    """Same species, or both humanoid. `left` and `right` are broadcastable index arrays."""
    return ((state.species_id[left] == state.species_id[right]) |
            (table.is_humanoid[state.params_id[left]] & table.is_humanoid[state.params_id[right]]))


def _take_slot(bucket, taken):
    while bucket:
        slot = bucket.pop()
        if not taken[slot]:
            taken[slot] = True
            return slot
    return None


def _join_existing_unions(state, table, rng, eligible, marriage_type, max_size):
    """Leftover singles join existing unions that still have room (polygyny / polyandry / polygamy).

    A non-humanoid only joins a union of its own species. Humanoids may join other humanoids.
    A xenophile who passes a coin flip prefers a union that already holds another species.
    """
    singles = eligible & (state.union_id < 0)
    herm = table.is_hermaphrodite[state.params_id]
    if marriage_type == "polygyny":
        singles = singles & (state.is_female | herm)
    elif marriage_type == "polyandry":
        singles = singles & (~state.is_female | herm)
    joiners = np.flatnonzero(singles)
    if len(joiners) == 0:
        return 0
    stats = UnionStats(state, table)
    if len(stats.labels) == 0:
        return 0
    space = np.maximum(0, max_size - stats.size)
    open_union = space > 0
    if marriage_type == "polygyny":
        open_union = open_union & stats.has_males
    elif marriage_type == "polyandry":
        open_union = open_union & stats.has_females
    open_idx = np.flatnonzero(open_union)
    if len(open_idx) == 0:
        return 0

    coupled = stats.coupled
    inv = stats.inverse
    member_species = state.species_id[coupled]
    count = len(stats.labels)
    sp_min = np.full(count, np.iinfo(np.int32).max, dtype=np.int32)
    sp_max = np.full(count, np.int32(-1), dtype=np.int32)
    np.minimum.at(sp_min, inv, member_species)
    np.maximum.at(sp_max, inv, member_species)
    uniform = np.where(sp_min == sp_max, sp_min, np.int32(-1))
    all_humanoid = np.bincount(inv, weights=(~table.is_humanoid[state.params_id[coupled]]).astype(np.int32),
                               minlength=count) == 0

    slot_label = np.repeat(stats.labels[open_idx], space[open_idx])
    slot_species = np.repeat(uniform[open_idx], space[open_idx])
    slot_humanoid = np.repeat(all_humanoid[open_idx], space[open_idx])
    if len(slot_label) == 0:
        return 0
    own_bucket = {}
    cross_bucket = {}
    mixed_bucket = []
    for slot in rng.permutation(len(slot_label)).tolist():
        species = int(slot_species[slot])
        if species >= 0:
            own_bucket.setdefault(species, []).append(slot)
        if slot_humanoid[slot] and species >= 0:
            cross_bucket.setdefault(species, []).append(slot)
        elif slot_humanoid[slot]:
            mixed_bucket.append(slot)

    taken = np.zeros(len(slot_label), dtype=bool)
    cross_species = list(cross_bucket.keys())
    wants_other = table.is_xenophilic[state.params_id[joiners]] & table.is_humanoid[state.params_id[joiners]]
    if wants_other.any():
        wants_other = wants_other & (rng.random(len(joiners)) < 0.5)
    assigned = 0
    for j in rng.permutation(len(joiners)).tolist():
        person = int(joiners[j])
        sid = int(state.species_id[person])
        human = bool(table.is_humanoid[state.params_id[person]])
        slot = None
        if wants_other[j]:
            slot = _take_slot(mixed_bucket, taken)
            if slot is None and cross_species:
                start = int(rng.integers(0, len(cross_species)))
                for step in range(len(cross_species)):
                    species = cross_species[(start + step) % len(cross_species)]
                    if species == sid:
                        continue
                    slot = _take_slot(cross_bucket[species], taken)
                    if slot is not None:
                        break
        if slot is None:
            slot = _take_slot(own_bucket.get(sid, []), taken)
        if slot is None and human:
            slot = _take_slot(mixed_bucket, taken)
            if slot is None:
                for species, bucket in cross_bucket.items():
                    if species == sid:
                        continue
                    slot = _take_slot(bucket, taken)
                    if slot is not None:
                        break
        if slot is not None:
            state.union_id[person] = slot_label[slot]
            assigned += 1
    return assigned


def couple_singles(state, table, rng, eligible, partner_pool, marriage_chance, sample_cap=20,
                   marriage_type="monogamy"):
    """
    Coupling phase. Each eligible single individual tries to pair with probability `marriage_chance`; an attempt
    looks at a random sample of up to `sample_cap` members of the partner pool and takes the one closest in age that is
    still single, not itself, and of a compatible sex (opposite sexes, or both hermaphrodites).

    The pool is fixed at the start of the phase (as before), but attempts are resolved in order, so a partner taken by
    an earlier attempt is no longer available to a later one. Samples and each attempter's candidates ranked by age
    difference are computed for all attempters at once; only the final in-order claim resolution is a Python loop, and
    it runs once per attempter, not once per individual.

    A pairing has to be the same species, or both humanoid. Non-humanoids never couple outside their species.
    A xenophile who passes a coin flip samples partners of another species first, and falls back to the usual
    closest-age partner when no other species is available.

    Non-monogamous marriage types then let leftover singles join existing unions that still have room: polygyny is
    1 male + many females (max 5), polyandry is 1 female + many males (max 5), polygamy is many + many (max 9).
    Warfare can still enlarge a union past these caps, including across a species line a non-humanoid would not choose.
    Returns the number of people newly placed into a union (two per new pair, plus joiners).
    """
    joined = 0
    max_size = MARRIAGE_MAX_SIZE.get(marriage_type, 2)
    if marriage_type != "monogamy":
        joined += _join_existing_unions(state, table, rng, eligible, marriage_type, max_size)

    still_eligible = eligible & (state.union_id < 0)
    still_pool = partner_pool & (state.union_id < 0)
    eligible_idx = np.flatnonzero(still_eligible)
    pool_idx = np.flatnonzero(still_pool)
    k = min(len(pool_idx), sample_cap)
    paired = 0
    if len(eligible_idx) > 0 and k > 0:
        attempters = eligible_idx[rng.random(len(eligible_idx)) < marriage_chance]
        if len(attempters) > 0:
            candidates = pool_idx[_distinct_samples(rng, len(attempters), len(pool_idx), k)]          # [attempters, k]
            herm = table.is_hermaphrodite[state.params_id]
            female = state.is_female
            compatible = ((herm[attempters][:, None] & herm[candidates]) | (female[attempters][:, None] != female[candidates]))
            species_ok = _species_compatible(state, table, attempters[:, None], candidates)
            usable = (candidates != attempters[:, None]) & compatible & species_ok
            age_gap = np.where(usable, np.abs(state.age_days[candidates] - state.age_days[attempters][:, None]), np.inf)
            ranked = np.argsort(age_gap, axis=1, kind="stable")        # closest first; ties keep sample order (first minimum)
            ranked_candidates = np.take_along_axis(candidates, ranked, axis=1).tolist()
            ranked_usable = np.isfinite(np.take_along_axis(age_gap, ranked, axis=1)).tolist()

            xen = table.is_xenophilic[state.params_id[attempters]] & table.is_humanoid[state.params_id[attempters]]
            if xen.any() and (wants_other := xen & (rng.random(len(attempters)) < 0.5)).any():
                pool_species = state.species_id[pool_idx]
                order = np.argsort(pool_species, kind="stable")
                sorted_pool = pool_idx[order]
                uniq, starts, counts = np.unique(pool_species[order], return_index=True, return_counts=True)
                start_of = {int(s): int(a) for s, a in zip(uniq, starts)}
                count_of = {int(s): int(c) for s, c in zip(uniq, counts)}
                n_pool = len(pool_idx)
                for row in np.flatnonzero(wants_other).tolist():
                    person = int(attempters[row])
                    sid = int(state.species_id[person])
                    own_n = count_of.get(sid, 0)
                    other_n = n_pool - own_n
                    if other_n <= 0:
                        continue
                    take = min(sample_cap, other_n)
                    picks = _distinct_samples(rng, 1, other_n, take)[0]
                    mapped = picks + own_n * (picks >= start_of.get(sid, 0))
                    alt = sorted_pool[mapped]
                    sex_ok = ((herm[person] & herm[alt]) | (female[person] != female[alt])) & (alt != person)
                    sex_ok = sex_ok & table.is_humanoid[state.params_id[alt]]
                    if not sex_ok.any():
                        continue
                    gaps = np.where(sex_ok, np.abs(state.age_days[alt] - state.age_days[person]), np.inf)
                    keep = np.argsort(gaps, kind="stable")
                    keep = keep[np.isfinite(gaps[keep])]
                    ranked_candidates[row] = alt[keep].tolist()
                    ranked_usable[row] = [True] * len(keep)

            taken = set()
            first, second = [], []
            for row, person in enumerate(attempters.tolist()):
                if person in taken:                                    # already paired earlier in this phase
                    continue
                for candidate, ok in zip(ranked_candidates[row], ranked_usable[row]):
                    if not ok:
                        break
                    if candidate not in taken:
                        taken.add(person)
                        taken.add(candidate)
                        first.append(person)
                        second.append(candidate)
                        break
            if first:
                labels = state.new_union_ids(len(first))
                state.union_id[first] = labels
                state.union_id[second] = labels
                paired = len(first)

    if marriage_type != "monogamy":
        joined += _join_existing_unions(state, table, rng, eligible, marriage_type, max_size)
    return paired + joined

# 4. reproduction

def age_based_fertility(age_years, max_fertility, rising_steepness, rising_midpoint, falling_steepness, falling_midpoint):
    """Vector form of calculate_age_based_fertility (double logistic). Overflow gives 0, as before."""
    with np.errstate(over="ignore"):
        growth = 1.0 / (1.0 + np.exp(-rising_steepness * (age_years - rising_midpoint)))
        decline = 1.0 - 1.0 / (1.0 + np.exp(-falling_steepness * (age_years - falling_midpoint)))
    return max_fertility * growth * decline

class ReproductionResult:
    def __init__(self, newborns, births, infant_deaths, eligible_mothers, gave_birth, piglet_calories):
        self.newborns = newborns
        self.births = births
        self.infant_deaths = infant_deaths
        self.eligible_mothers = eligible_mothers   # "Potential Mothers" in the report
        self.gave_birth = gave_birth               # bool mask over the population: mothers who conceived this cycle
        self.piglet_calories = piglet_calories

def _find_fathers(state, table, rng, mothers, stats, partner):
    """father[j] = index of the father of mothers[j], or -1 when the mother has no male-role partner (asexual rule).

    Also returns cross_blocked: a partner of another species exists, but a non-humanoid makes the cross illegal,
    and no compatible male in the union can father the child. Those mothers must not fall through to asexual birth.
    """
    herm = table.is_hermaphrodite[state.params_id]
    male_like = ~state.is_female | herm
    other = partner[mothers]
    has_pair_partner = other >= 0
    safe_other = np.where(has_pair_partner, other, 0)
    pair_ok = has_pair_partner & _species_compatible(state, table, mothers, safe_other)
    father = np.where(pair_ok & male_like[safe_other], other, -1)
    cross_blocked = has_pair_partner & ~pair_ok
    big = stats.per_individual(stats.size, len(state), default=0)[mothers] > 2     # harems
    for j in np.flatnonzero(big):
        members = np.flatnonzero(state.union_id == state.union_id[mothers[j]])
        options = [m for m in members if m != mothers[j] and male_like[m]
                   and _species_compatible(state, table, mothers[j], m)]
        if options:
            father[j] = options[int(rng.integers(len(options)))]
            cross_blocked[j] = False
        else:
            father[j] = -1
            cross_blocked[j] = True
    return father, cross_blocked

def reproduce(state, table, names, rng, current_day, age_years, genetic_adjuster, plague_count, pollution_count,
              make_hybrid_row, hybridization_type="codominant"):
    """
    Reproduction phase over every female who is fertile, in the fertile ages, in a viable union and due to breed.

    Per eligible mother: contraception draw, age-based fertility (times the father's age-based fertility), conception
    draw, number of births (multiple births), then for each child: whether it survives infancy, its sex, whether it is
    fertile and, for backwards-ageing species, its starting age.

    Hybridization (see README): only two humanoids of different species can hybridize. A non-humanoid breeds only
    with its own species; a cross that warfare or a glitch created produces no children. Dominant species always
    produce full members of that species. Otherwise `codominant` (default) yields full males and females of either
    parent species, `midpoint` blends parameters (`make_hybrid_row` / SimulationParameters.from_parents), and
    `lineal` gives daughters the mother's species and sons the father's.

    Differences from the object-based loop (all only matter for polygynous harems, which only warfare creates, or
    for hermaphrodite pairs): a father's "max children" check uses his count from the start of the phase rather than a
    running count; the random draws themselves are, of course, different draws.
    Modifies `state.children` and `state.next_breeding_day` in place.
    """
    n = len(state)
    p = state.params_id
    age_days_equiv = age_years * E
    stats = UnionStats(state, table)
    viable_union = (stats.size >= 2) & stats.has_females & stats.has_males
    in_viable_union = stats.per_individual(viable_union, n)
    eligible = (state.is_female & state.is_fertile & (state.next_breeding_day <= current_day) & in_viable_union
                & (table.puberty_age_days[p] <= age_days_equiv) & (age_days_equiv < table.menopause_age_days[p])
                & (state.children < table.max_kids_per_primate[p]))
    mothers = np.flatnonzero(eligible)
    no_births = ReproductionResult(state.take(np.zeros(n, dtype=bool)), 0, 0, len(mothers), np.zeros(n, dtype=bool), 0)
    if len(mothers) == 0:
        return no_births
    M = len(mothers)
    mp = p[mothers]
  
    contraceptive = rng.random(M) < table.contraception_abortion_use_rate[mp]   # ---- conception ----
    peak = table.effective_per_cycle_fertility_rate[mp]
    fertility = np.where(table.flat_fertility[mp], peak, age_based_fertility(
        age_years[mothers], peak, table.fertility_rising_steepness[mp], table.fertility_rising_midpoint[mp],
        table.fertility_falling_steepness[mp], table.fertility_declining_midpoint[mp]))

    partner = stats.partners_of_pairs(n)
    father, cross_blocked = _find_fathers(state, table, rng, mothers, stats, partner)
    has_father = father >= 0
    fi = np.where(has_father, father, 0)
    fp = p[fi]
    father_maxed = has_father & (state.children[fi] >= table.max_kids_per_primate[fp])    # both parents stop at the cap
    father_age_days = age_in_days_roundtrip(state.age_days[fi])
    lifespan = table.lifespan_days[fp]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        male_fertility = np.where(lifespan > 0, 1.0 / (1 + np.exp(10 * (father_age_days / np.where(lifespan > 0, lifespan, 1) - 0.8))), 0.01)
    fertility = fertility * np.where(has_father, male_fertility, 1.0)
    fertility = np.where(contraceptive, fertility * 0.123, fertility)
    conceives = (rng.random(M) <= np.maximum(0, fertility)) & ~father_maxed & ~cross_blocked
    c = np.flatnonzero(conceives)                         # positions within `mothers`
    gave_birth = np.zeros(n, dtype=bool)
    gave_birth[mothers[c]] = True
    if len(c) == 0:
        no_births.gave_birth = gave_birth
        return no_births

    mom, mrow = mothers[c], mp[c]
    dad, drow, with_dad = father[c], fp[c], has_father[c]
    chance_multiple = table.chance_of_multiple_birth[mrow]
    with np.errstate(divide="ignore", invalid="ignore"):   # extra births ~ geometric: P(at least k more) = chance ** k
        extra = np.floor(np.log(1.0 - rng.random(len(c))) / np.log(chance_multiple))
    births_each = 1 + np.where(chance_multiple > 0, extra, 0).astype(np.int64)

    # ---- which parameters does the child use? (hybrid rules) ----
    is_hybrid = with_dad & (state.species_id[mom] != state.species_id[np.where(with_dad, dad, 0)])
    child_row = np.where(with_dad & (rng.random(len(c)) >= 0.5), drow, mrow)          # codominant: full either species
    mom_dominant, dad_dominant = table.is_dominant[mrow], table.is_dominant[drow]
    aborted = is_hybrid & mom_dominant & dad_dominant                                   # two dominant species: non-viable
    takes_mother = is_hybrid & mom_dominant & ~dad_dominant
    takes_father = is_hybrid & ~mom_dominant & dad_dominant
    child_row = np.where(takes_mother, mrow, np.where(takes_father, drow, child_row))
    is_hybrid = is_hybrid & ~takes_mother & ~takes_father
    blendable = is_hybrid & ~aborted
    if hybridization_type == "midpoint":
        for j in np.flatnonzero(blendable):                                            # blended parameters
            child_row[j] = make_hybrid_row(int(mrow[j]), int(drow[j]))
    elif hybridization_type == "lineal":
        pick_ratio = np.where(rng.random(len(c)) < 0.5, table.sex_ratio_at_birth[mrow], table.sex_ratio_at_birth[drow])
        lineal_daughter = rng.random(len(c)) <= pick_ratio
        child_row = np.where(blendable, np.where(lineal_daughter, mrow, drow), child_row)
    births_each = np.where(aborted, 0, births_each)
    breeding_done = c[~aborted]
    state.next_breeding_day[mothers[breeding_done]] = current_day + table.interbirth_interval_days[mp[breeding_done]]

    # ---- infant survival ----
    infant_mortality = table.infant_mortality_rate[child_row] * (1.0 + (1.0 - genetic_adjuster)) ** 1.59
    infant_mortality = infant_mortality / table.genetic_diversity[mrow]
    infant_mortality = np.where(is_hybrid & with_dad, infant_mortality / table.genetic_diversity[drow], infant_mortality)
    infant_mortality = infant_mortality + 0.2 * plague_count
    pollution_sterility = 0.7 * pollution_count #now a cycle with no breeding population simply loses every infant.
    threshold = infant_mortality / genetic_adjuster if genetic_adjuster > 0 else np.full(len(c), np.inf)

    mother_of = np.repeat(np.arange(len(c)), births_each)       # one entry per child born alive or dead this cycle
    survives = rng.random(len(mother_of)) > threshold[mother_of]
    born = mother_of[survives]
    dead = mother_of[~survives]
    births = len(born)

    crow = child_row[born] # ---- the surviving children ----
    female_child = np.where(table.is_hermaphrodite[crow], True, rng.random(births) < table.sex_ratio_at_birth[crow])
    if hybridization_type == "lineal":
        female_child = np.where(is_hybrid[born], crow == mrow[born], female_child)
    sterile_chance = table.sterile_chance[crow] + pollution_sterility
    hybrid_boost = np.where(female_child, 0.2, 0.4)           # males of hybrid parentage are more often sterile
    sterile_chance = sterile_chance + np.where(with_dad[born] & is_hybrid[born], hybrid_boost, 0.0)
    fertile_child = rng.random(births) > sterile_chance
    lifespan_c = table.lifespan_days[crow]
    sigma = 0.175   # lets about 1 in 10,000 backwards-ageing births start near the maximum age
    mu = np.log(np.where(lifespan_c > 0, lifespan_c, 1.0)) - 0.5 * sigma ** 2
    start_age = np.where(table.ages_backward[crow] & (lifespan_c > 0),
                         np.floor(rng.lognormal(mu, sigma)), 0.0)
    newborns = state.newborns(state, births, start_age, female_child, table.name_id[crow], crow, fertile_child)

    kids = np.bincount(born, minlength=len(c))
    state.children[mom] += kids
    np.add.at(state.children, dad[with_dad], kids[with_dad])

    # ---- infants that died ----
    piglets = 0
    if len(dead):
        piglets = 1000 * int(np.count_nonzero(with_dad[dead] & (table.kid_fodder[mrow[dead]] | table.kid_fodder[drow[dead]])))
        revive = table.respawn_as_male[mrow[dead]] & state.is_female[mom[dead]]
        if revive.any():
            source = dead[revive]
            newborns = newborns.append(respawned_males(state, len(source), mrow[source], state.species_id[mom[source]], table, rng))
    return ReproductionResult(newborns, births, len(dead), len(mothers), gave_birth, piglets)

# 5. maternal and adult mortality

def final_deaths(state, table, names, rng, age_years, gave_birth, adult_mortality_scale, plague_count, cycle_length_years):
    """
    Maternal mortality (for mothers who conceived this cycle) and adult mortality (everyone older than half a year).
    A mother who survives childbirth still faces adult mortality. Returns (died mask, tally, respawned males).
    """
    n = len(state)
    p = state.params_id
    died_in_childbirth = gave_birth & (rng.random(n) <= table.maternal_mortality_rate[p])
    adult_mortality = table.adult_mortality_rate[p] * cycle_length_years * adult_mortality_scale
    for _ in range(plague_count):
        adult_mortality = adult_mortality + 0.2
    died_as_adult = ~died_in_childbirth & (age_years > 0.5) & (rng.random(n) < adult_mortality)
    died = died_in_childbirth | died_as_adult
    tally = count_adult_deaths(state, died, table)
    revive = died & state.is_female & table.respawn_as_male[p]
    count = int(np.count_nonzero(revive))
    respawned = respawned_males(state, count, p[revive], np.full(count, names.id(DOUBLES_NAME)), table, rng)
    return died, tally, respawned

# 6. feeding

def greedy_take(needs, available):
    """
    Exact vector form of the sequential rule "in order, take your need from the pool if it is still large enough":

        for need in needs:  if available >= need: available -= need; taken

    Note that an individual who does not fit is skipped without ending the process, so smaller needs later in the
    order can still be met. Each round takes the longest run of items that fit (a cumulative sum), skips the first
    item that does not, and then drops every item larger than what is left (the pool only shrinks, so those can never
    fit later). A round removes at least one distinct need size, so only a handful of rounds are needed.
    """
    taken = np.zeros(len(needs), dtype=bool)
    remaining = float(available)
    candidates = np.arange(len(needs))
    while candidates.size:
        running = np.cumsum(needs[candidates])
        fit = int(np.searchsorted(running, remaining, side="right"))      # leading items that all fit
        if fit:
            taken[candidates[:fit]] = True
            remaining -= running[fit - 1]
        candidates = candidates[fit + 1:]                                  # candidates[fit] did not fit: skipped
        candidates = candidates[needs[candidates] <= remaining]
    return taken


def feed(state, table, rng, cycle_days, avail_meat, avail_veg, avail_grass, avail_water):
    """
    Food allocation. Everyone is put in a random order, then takes their need from the pool(s) their diet allows
    while it lasts: first come, first served, exactly as before.

    Diets and pools: carnivore->meat; herbivore->vegetation; ruminant->grass; omnivore->vegetation, else meat;
    "everything"->grass (independently), then vegetation, else meat; autotroph->water. Aquatic individuals
    additionally drink `cycle_days` of water each and are fed only if the water has not run out. Because each pool's
    outcome depends only on who reached it, the pools can be allocated one after another.

    Returns (shuffled state, fed mask over the shuffled state). The shuffled order replaces the population order.
    """
    n = len(state)
    shuffled = state.take(rng.permutation(n))
    p = shuffled.params_id
    need = caloric_need(shuffled, table) * cycle_days
    diet = table.diet_code[p]
    aquatic = table.is_aquatic[p]

    def pool(candidate_mask, available):
        idx = np.flatnonzero(candidate_mask)
        got = np.zeros(n, dtype=bool)
        got[idx[greedy_take(need[idx], available)]] = True
        return got

    fed_grass = pool((diet == 3) | (diet == 5), avail_grass)
    fed_veg = pool((diet == 2) | (diet == 4) | (diet == 5), avail_veg)
    fed_meat = pool((diet == 1) | (((diet == 4) | (diet == 5)) & ~fed_veg), avail_meat)    # those who missed vegetation
    fed = fed_grass | fed_veg | fed_meat

    water_users = np.flatnonzero((diet == 0) | aquatic)       # few or none: a plain loop keeps the water rules readable
    water = float(avail_water)
    for i in water_users.tolist():
        if diet[i] == 0 and water >= need[i]:
            water -= need[i]
            fed[i] = True
        if aquatic[i]:
            water -= cycle_days
            fed[i] = water >= 0
    return shuffled, fed

# 7. warfare (only reachable when disasters are enabled)

def combatants_of(state, table):
    """Males who are past puberty and not elderly."""
    p = state.params_id
    age_d = age_in_days_roundtrip(state.age_days)
    return np.flatnonzero(~state.is_female & (age_d >= table.puberty_age_days[p]) & (age_d < table.lifespan_days[p]))

def resolve_warfare(state, table, rng, combatants):
    """
    Battle royale: combatants are paired off at random, one of each pair dies (an odd one out survives automatically)
    and the winner takes the loser's female partners into his own union. Returns (survivor indices, killed indices).
    """
    print("   -> WARFARE BREAKS OUT (Battle Royale)")
    order = rng.permutation(combatants).tolist()
    survivors, killed = [], []
    limit = len(order)
    if limit % 2: 
        survivors.append(order[-1])
        limit -= 1
    herm = table.is_hermaphrodite[state.params_id]
    members = {}
    for i in np.flatnonzero(state.union_id >= 0).tolist():
        members.setdefault(int(state.union_id[i]), []).append(i)
    for k in range(0, limit, 2):
        first, second = order[k], order[k + 1]
        winner, loser = (first, second) if rng.random() < 0.5 else (second, first)
        survivors.append(winner)
        killed.append(loser)
        label = int(state.union_id[loser])
        if label >= 0:   # spoils of war: the loser's female partners move to the winner
            spoils = [m for m in members[label] if m != loser and (state.is_female[m] or herm[m])]
            if spoils:
                if state.union_id[winner] < 0:
                    state.union_id[winner] = state.new_union_ids(1)[0]
                    members[int(state.union_id[winner])] = [winner]
                for m in spoils:
                    members[label].remove(m)
                    state.union_id[m] = state.union_id[winner]
                    members[int(state.union_id[winner])].append(m)
            members[label].remove(loser)
            state.union_id[loser] = state.NO_UNION
    print(f"     Combatants: {len(order)}")
    print(f"     Casualties: {len(killed)}")
    return survivors, killed
