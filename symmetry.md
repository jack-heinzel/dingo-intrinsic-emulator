# Using a mirror symmetry to learn a density

This note explains one idea. Some quantities we want to learn are unchanged when we "mirror" the inputs, and that
fact can be built directly into training and evaluation. It starts with a one-dimensional toy and then applies the
same steps to the binary black hole problem.

---

## 1. The problem: densities that stop at a wall

Suppose a quantity *x* can only be positive, and we want a computer model to learn its probability density from
examples. Most flexible density models describe smooth densities on the whole number line. They have no built-in
notion of a wall at *x* = 0.

If the true density is still large right at the wall, the model has two bad options:

- drop very steeply to zero at the wall, which is hard to learn, or
- spill some probability onto the forbidden side (*x* < 0). That probability is wasted, and the density just inside
  the wall comes out too low.

Either way, the answer is worst exactly at the wall. In our problem, the walls are often physically interesting
places, such as equal masses or zero spin.

## 2. The fix in one dimension: unfold, learn, fold back

Suppose we also know that the quantity behind the density doesn't care about the sign of *x*: values at +*x* and
−*x* would be equally likely. Then we can:

1. **Unfold.** For each training example *x*, flip a fair coin. On heads use *x*; on tails use −*x*. The model now
   sees a density on the whole line that is the mirror image of itself, with half of the original on each side.
   There is no wall any more: the density passes smoothly through zero.
2. **Learn** this symmetric, wall-free density with the usual method.
3. **Fold back.** To get the density at a positive *x*, add the model's values at the point and at its mirror image:

   p(*x*) = q(*x*) + q(−*x*).

Why adding is correct: any probability the model put at −*x* really belongs to *x*, because those two points describe
the same thing. Folding returns it to where it belongs, so nothing is lost.

Two useful facts follow.

- **The folded density always integrates to exactly one.** The total over the positive side of q(*x*) + q(−*x*) equals
  the total of q over the whole line, which is one. This holds even if the model learned an imperfect, slightly
  lopsided density. The fold doesn't need the model to be perfect to give a properly normalized answer.
- **The folded density is flat at the wall.** Mirror images meet at zero, so the folded density has zero slope there.
  This is the "reflective boundary condition": probability bounces off the wall instead of leaking through it.

Sampling works the same way in reverse: draw from the model, then replace each negative draw by its mirror image.

## 3. The coin flip doesn't need to be a real symmetry of the data

In the binary black hole problem, the network learns a density for the parameters *given observed data*. The training
examples are pairs (parameters, data).

When we flip a parameter label with a coin, we leave the data alone. The coin is independent of the data, so the data
says nothing about how the coin landed. The best the model can do is give both labelings equal weight. The symmetry
holds by construction. We don't need to assume or check that any physical process is symmetric.

This only works when the two labels really describe *the same physical situation*, so that folding them back
together loses nothing. The next two sections show two such cases.

## 4. Case 1: which object do we call "number 1"?

A binary has two black holes. By convention the heavier one is "object 1" and the lighter one "object 2". This is
only a naming choice: calling the lighter one "object 1" describes the same binary.

Because of the convention, the mass ratio (lighter over heavier) can't exceed one, so there is a wall at **equal
masses**. Many binaries sit close to equal masses, so this wall matters.

The fix is the same unfold–learn–fold idea:

- Describe the mass ratio by its logarithm. Swapping the names of the two objects then just flips its sign, a mirror
  image about zero, like *x* → −*x* above.
- When the names are swapped, **everything** attached to the objects swaps with them: object 1's spin becomes object
  2's spin and vice versa. Swapping only the masses would describe a different binary.
- Overall size (the "chirp mass") and distance don't depend on the names, so they are unchanged.

## 5. Case 2: describing a spin without the "zero spin" problem

Each black hole's spin has a size *a* (from 0 to 1) and a tilt angle *θ* (how far it leans away from the orbit's
axis). Describing it by the pair (*a*, cos *θ*) has a hidden flaw.

**At zero spin, the angle means nothing.** A spin of size zero doesn't point anywhere, so every tilt describes the
same black hole. The quantity we want to learn (how well a spin explains the data) must therefore give the same
answer for every tilt when *a* = 0. In the (*a*, cos *θ*) description, that single point is spread out into a whole
line of tilts, and nothing forces a model to treat them equally.

**Use "arrow" coordinates instead.** Draw the spin as an arrow in a flat plane:

- the **along-axis** part, *a* cos *θ*, the component along the orbit's axis;
- the **sideways** part, *a* sin *θ*, the component perpendicular to it.

Zero spin is now a single point, the origin of the plane. Any density that is smooth there automatically gives every
tilt the same value as the arrow shrinks to zero.

This leaves two walls, and each gets its own fix.

- **The sideways wall.** As defined, the sideways part is never negative. But the sideways part also has a
  direction around the axis, which we average over. Pointing the sideways part "left" or "right" describes equally
  likely situations. So we let the sideways part take either sign and use a coin flip, exactly as in Section 2.
  Averaging around the axis makes the density mirror-symmetric across the axis, so unfold–learn–fold applies.
- **The outer wall at *a* = 1.** Spins can't exceed 1, so the arrow lives inside a disk of radius 1, with a circular
  wall. We remove it by stretching the radius: keep the arrow's direction, but replace its length *a* by a stretched
  length that grows without limit as *a* approaches 1. (The function used is the inverse hyperbolic tangent. It
  behaves like *a* itself near zero, so the origin stays smooth.) The disk becomes the whole plane, with no wall left.

## 6. Putting the pieces together

There are three independent coin flips, each mirroring something:

| Operation | What changes |
|---|---|
| swap the names of the two objects | sign of the log mass ratio; the two spin arrows trade places |
| flip object 1's sideways spin | sign of object 1's sideways part |
| flip object 2's sideways spin | sign of object 2's sideways part |

Combining these in every possible way gives **8** distinct relabelings, including "change nothing". (Order can matter: swapping the names and then flipping
object 1's spin is the same as flipping object 2's spin and then swapping. However they are combined, there are
exactly 8 distinct outcomes.) Every relabeling just mirrors or reorders coordinates. None stretches or shrinks space, which is
why the totals in the folding argument match exactly.

The recipe:

1. **Training.** Simulate each example with the conventional labels: heavier object first, sideways spins
   non-negative. Then pick one of the 8 relabelings at random and apply it to the parameters only. The simulated
   data are untouched.
2. **Evaluating the density** at a conventionally labeled point: apply all 8 relabelings, evaluate the model at each
   of the 8 resulting points, and add the values. The sum is properly normalized over the conventional region, for
   the same reason as in Section 2.
3. **Sampling:** draw from the model, then undo whichever relabeling brings each draw back to the conventional
   labels.

The cost is 8 model evaluations instead of 1 for each density evaluation. These run in parallel, so in practice the
cost is small.

## 7. A note on the starting distribution of the training examples

Training examples are drawn from a chosen starting distribution (the "prior") over parameters. Two points matter
here.

- **The starting distribution must respect the same symmetry.** It must give equal weight to all 8 relabelings of a
  point. Otherwise, the coin flips and the starting distribution would disagree. Distributions that treat the two
  objects alike, and that depend on a spin only through its size and the size (not sign) of its tilt, satisfy this.
- **Its shape in the new coordinates matters.** A common choice is uniform in spin size *a* and uniform in cos *θ*.
  Redrawn in the arrow plane, this distribution has a sharp spike at the origin and a crease along the axis. A
  learned density has to reproduce both, and the spike sits at zero spin, exactly where we wanted accuracy.

  The model's output is ultimately divided by the starting distribution, so the training distribution doesn't have
  to be the one used for the final scientific answer. One option is to train with a distribution that is smooth in
  the arrow plane, then convert to the uniform-in-*a*, uniform-in-cos *θ* choice afterwards. The conversion
  multiplies by the ratio of the two distributions, which is known exactly.
