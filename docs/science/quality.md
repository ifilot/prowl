# Assess projection quality

A bonding analysis is only as good as the local picture it is based on. The
quality report tells you how well that picture describes your VASP calculation.
Read it before interpreting bond populations. Each diagnostic tests a different
part of the calculation, and no single number establishes accuracy on its own.
The CLI prints the report after `run`, and Python exposes `result.quality_report()`.

## A quick checklist

For each analysis, ask:

1. **Is the PAW norm deviation tiny?** If not, the wavefunctions and potentials
   do not match. Fix the inputs before anything else.
2. **Is the charge spilling low?** If not, the basis misses part of the occupied
   states. Consider polarization shells or another basis.
3. **Do the electron counts agree?** The DFT, Löwdin and Mulliken electron
   counts should be close to one another.
4. **Is the window sensitivity small?** If not, your bond populations depend on
   a technical choice. Report it, and check the band count and basis.
5. **Are there warnings?** In particular, check whether any of your bonds cannot
   be resolved on this k-mesh.

The table below explains each diagnostic in more detail.

## What the diagnostics measure

| Diagnostic | What to examine | Useful next action |
| --- | --- | --- |
| Maximum PAW norm deviation | Consistency of wavefunctions and reconstructed PAW normalization | Verify the POTCAR, electronic state, LREAL and wavefunction encoding. |
| Absolute occupied charge spilling | Occupied-state weight missed by the chosen local projection | Inspect the basis and compare physically justified additional shells. |
| Absolute total spilling | Representation of the retained states, including unoccupied character | Compare bases and window choices; keep the retained range in mind. |
| Minimum overlap eigenvalue | Near-linear dependence of local orbitals | Reconsider redundant or overly diffuse functions. |
| Maximum occupied reconstruction residual | How well the projected representation reconstructs occupied band energies | Check the window, basis and consistency of the analysis. |
| DFT / Löwdin / Mulliken electron counts | Accounting of electrons across the projection and populations | Investigate discrepancies alongside spilling and conventions. |
| Window sensitivity | Change in bond populations under the alternative window | Report important sensitivity and examine basis/band convergence. |
| Periodic-image aliasing | Whether a selected bond is distinguishable on this k-mesh | Increase the mesh or shorten the selected range. |

## Charge spilling

For every band, PROWL compares the norm of the wavefunction with the norm of its
projection onto the local orbitals. The difference is the part of the state the
basis cannot represent. Averaged over the occupied states, with absolute values
so that errors cannot cancel, this gives the **occupied charge spilling**:

$$
S_\text{occ} = \frac{\sum_{\mathbf{k}} w_\mathbf{k} \sum_n f_{n\mathbf{k}}\,
\left|1 - \langle\psi_{n\mathbf{k}}|\hat P_\text{local}|\psi_{n\mathbf{k}}\rangle\right|}
{\sum_{\mathbf{k}} w_\mathbf{k} \sum_n f_{n\mathbf{k}}},
$$

where $\hat P_\text{local}$ projects onto the local orbitals. The **total
spilling** is the same average taken over all retained bands without
occupation weights. It is usually larger, because unoccupied bands are harder
to describe with atomic orbitals. The report prints both as percentages.

## Implementation thresholds are not acceptance criteria

The default PAW norm tolerance is `1e-4`. With strict validation enabled, a
deviation above it stops analysis. The default check samples bands (every eighth
band and the last band per spin/k-point for the folded path) and escalates to a
full check when the sampled deviation exceeds half the tolerance. Use
`numerical.paw_norm_check: full` when you need every band checked; the report
records the checking mode.

When occupied charge spilling exceeds 3%, PROWL may suggest available polarization
shells. That suggestion threshold is not a universal acceptable-error boundary.
A small residual or small spilling value can coexist with window sensitivity or
an unsuitable chemical representation.

The relative window-sensitivity summary considers bonds with `|ICOHP| > 0.1 eV`
to avoid ratios dominated by near-zero denominators. Inspect the alternate
values as well. An unavailable alternative is not a zero difference.

## A practical convergence sequence

1. Establish consistent inputs and a sufficiently converged underlying DFT state.
2. Converge the k-point mesh for the distances and real-space pairs of interest.
3. Supply sufficient reliable bands for the selected basis and window repair.
4. Compare reasonable basis choices, including polarization where warranted.
5. Check whether the chemical conclusion persists under window and basis changes.
6. Record the chosen settings, diagnostics, sensitivity and limitations.

Change one aspect at a time when diagnosing a discrepancy. Enlarging the basis
changes both orbital flexibility and the required band count; increasing NBANDS
can change a frozen complement. Neither adjustment is automatically an
improvement without checking the resulting diagnostics.
