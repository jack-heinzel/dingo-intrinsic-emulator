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

This leaves two walls, and they need different treatment.

- **The sideways wall.** As defined, the sideways part is never negative. But the sideways part also has a
  direction around the axis, which we average over. Pointing the sideways part "left" or "right" describes equally
  likely situations. So we let the sideways part take either sign and use a coin flip, exactly as in Section 2.
  Averaging around the axis makes the density mirror-symmetric across the axis, so unfold–learn–fold applies.
- **The outer wall at *a* = 1.** Spins can't exceed 1, so the arrow lives inside a disk of radius 1. There is no
  mirror symmetry here: nothing physical says the density should be flat at *a* = 1. So we keep it as a real wall
  and make the model able to handle walls (Section 5b).

### 5a. Why not stretch the wall away?

A tempting fix is to stretch the disk until it covers the whole plane: keep the arrow's direction, and replace its
length by one that grows without limit as *a* approaches 1. The wall disappears, but this is fragile.

Near the old wall, a tiny range of *a* is stretched over an infinite range of the new length. The density at *a* = 1
then depends entirely on how fast the model's density dies off far away, in its "tail". If the tail falls slightly
too slowly, the density at *a* = 1 shoots up to infinity. If it falls slightly too fast, the density drops to zero.
Only one exact rate gives a sensible finite value, and a learned model has no reason to hit it. (The models we use
have tails that fall very fast, so they would push the density at *a* = 1 to zero every time.) The same problem
comes with any trick that sends a wall off to infinity.

Mirroring (Section 2) doesn't have this problem. It is a local reflection and never involves the far-away tail.

### 5b. Instead: turn the disk into a square, and use a model built for boxes

**Two kinds of coordinate trouble.** Before choosing how to turn the disk into a square, it helps to separate two
problems a coordinate change can have, because only one of them hurts us.

1. **One physical situation, many coordinate values.** This was the flaw of (*a*, cos *θ*) at zero spin: a whole
   line of coordinates described one black hole. The true answer must be equal along that line, but a learned model
   doesn't know it and can disagree with itself. **This is the trouble we must avoid.**
2. **Space squeezed or stretched at some points.** The coordinates are still one-to-one, but some regions are
   compressed. For a *density* this matters, because densities change when space is squeezed. But the thing we
   finally want, how well each parameter value explains the data, is not a density. It is an ordinary value
   attached to each physical situation, and it doesn't change when we relabel coordinates. We compute it as the
   ratio of two densities written in the *same* coordinates (the learned one and the starting one), so any squeeze
   affects both equally and cancels. Squeezing only changes where training examples land.

**A first attempt that fails.** At a given along-axis value, the sideways part can be at most √(1 − along²). Dividing
by that maximum fills a square, but at along = ±1 (full-size spin pointing exactly along or against the axis) a whole
edge of the square describes a single physical spin. That is trouble of kind 1, moved from zero spin to the two
poles.

**A map with no trouble of kind 1.** Name the square's coordinates *u* and *v*, each between −1 and 1, and set

  along-axis part = *u* √(1 − *v*²/2),  sideways part = *v* √(1 − *u*²/2).

This sends the square onto the disk **one-to-one**, including the edges: each edge of the square becomes a quarter of
the circle, and no edge is collapsed to a point. A square has corners and a circle doesn't, so some squeezing is
unavoidable. Here it happens only at the four corners of the square (trouble of kind 2, which cancels). We checked
these properties numerically. The map also keeps everything we wanted:

- Zero spin is the single point at the centre of the square.
- It is smooth everywhere, so the quantity we learn stays smooth.
- The sideways mirror is just *v* → −*v*.

**Going back from the disk to the square.** Write *x* for the along-axis part and *y* for the sideways part (with
its sign), so *x*² + *y*² ≤ 1. Then

  *u* = ½ √(2 + *x*² − *y*² + 2√2 *x*) − ½ √(2 + *x*² − *y*² − 2√2 *x*),

  *v* = ½ √(2 − *x*² + *y*² + 2√2 *y*) − ½ √(2 − *x*² + *y*² − 2√2 *y*).

*Why this works.* Squaring the forward formulas gives *x*² = *u*² − *u*²*v*²/2 and *y*² = *v*² − *u*²*v*²/2, so
*x*² − *y*² = *u*² − *v*². Call *B* = √(2 − *v*²); the forward formula says √2 *x* = *u B*. Then the quantity under
each square root in the formula for *u* is

  2 + *x*² − *y*² ± 2√2 *x* = *u*² ± 2*uB* + (2 − *v*²) = (*B* ± *u*)².

Since *B* is at least 1 and *u* is at most 1 in size, both *B* + *u* and *B* − *u* are non-negative, so the square
roots are exactly *B* + *u* and *B* − *u*. Half their difference is *u*. The formula for *v* works the same way, with
the roles of *u* and *v* swapped.

*Two practical notes for computing it.*

- At the four points on the circle where the square's corners land (*x* and *y* both ±1/√2), one of the quantities
  under a square root is exactly zero. Rounding can make it a tiny negative number, so clip it at zero before taking
  the root.
- Near the centre, the formula subtracts two nearly equal numbers, which loses accuracy. Multiplying top and bottom
  by the sum of the two roots gives an equivalent form that is accurate everywhere:

  *u* = 2√2 *x* / [√(2 + *x*² − *y*² + 2√2 *x*) + √(2 + *x*² − *y*² − 2√2 *x*)],

  and the same for *v* with *x* and *y* swapped (including the sign of the *x*² − *y*² term).

We checked numerically that going from the square to the disk and back returns the starting point, to about 4 × 10⁻¹⁴,
everywhere on the square including its edges and corners.

**A model built for boxes.** Every coordinate we learn now lives in a box with flat walls. (The overall mass and
distance coordinates already did, because their starting distributions have hard upper and lower limits.) We use a
density model that maps the box onto itself, built from flexible one-dimensional curves that start at one wall and
end at the other. With such a model:

- no probability can spill outside the box, and
- the density right at each wall is an ordinary finite number that the model learns from the data, rather than
  something fixed by far-away tail behaviour.

The mirrors from Section 2 still apply on top of this, at equal masses and at *v* = 0, where they make the density
flat across the mirror, as the physics requires. At the real walls (spin size 1, and the limits on mass and
distance) the model is free to learn whatever finite value the data support.

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
  Redrawn in the square coordinates, this distribution has a sharp spike at the centre (zero spin) and a crease
  along *v* = 0. A learned density has to reproduce both, and the spike sits at zero spin, exactly where we wanted
  accuracy.

  The model's output is ultimately divided by the starting distribution, so the training distribution doesn't have
  to be the one used for the final scientific answer. The simplest option is to train with examples spread
  **uniformly over the square**. That distribution is smooth, has no spike, and respects all 8 relabelings. We then
  convert to the uniform-in-*a*, uniform-in-cos *θ* choice afterwards. The conversion multiplies by the ratio of the
  two distributions, which is known exactly.

## 8. Update (November 15): keep the disk, drop the square

The square in section 5b was there for one reason: a model whose outputs can never leave a box. The models we
actually trained don't have that property. They start from a bell curve over all real numbers and can put a little
probability anywhere. To them, a square is no more natural a shape than a disk. So the square buys nothing, and the
next run (run C) describes each spin directly by its two arrow components, the "up" part and the "sideways" part,
inside the disk.

Everything that mattered survives:
- Zero spin is still a single point, so there is still no "zero spin" problem (section 5).
- The coin flip on the sign of the sideways part is unchanged (section 3), and so is the label swap (section 4).
- With no square map there is no stretching factor at all. Folding back works exactly as before.

What remains is the leak at the outer edge of the disk (spin magnitude 0.99). It is small (about 0.2% of the
model's samples) and the same as for the other runs.

(Run D replaced run C before C was trained: see the note at the end of this section.)

The training distribution changes too (section 7). Spread evenly over the disk, very few training examples have
small spins: only about 1 in 100 has magnitude below 0.1. Run C mixes two smooth pieces half and half: even over the
disk, plus a narrow round bump centred on zero spin. About 1 in 5 examples is then below 0.1. Both pieces are smooth
and treat every direction the same way, so there is still no spike for the model to learn, and the conversion to
the final uniform-in-*a*, uniform-in-cos *θ* choice is still an exact multiplication afterwards.

**Run D's training distribution.** Instead of the bump, run D trains with spin magnitudes spread evenly, like the
final choice we actually want, except very close to zero. Spread evenly in magnitude means the density *in the disk*
grows like 1/(magnitude) towards the centre, which is a spike. So inside a small core (magnitude below about 0.05)
the density is softened. The density in the disk is 1/√(magnitude² + 0.05²): equal to the spike far from the centre,
and a gentle rounded peak at the centre. Training then matches the final distribution almost everywhere, and the
model never has to learn a spike.
