Perceptron Learning Algorithm

## **1. Purpose & Intuition**
The perceptron is one of the earliest and simplest algorithms for supervised classification.

It finds a linear decision boundary that separates two classes.

Goal: Find weight w and bias b such that:

sign(w·x + b) = y

for all training points (x,y) where y ∈ {+1, −1}.

Core idea: Start with some guess for w and b, then iteratively nudge them when we misclassify a point.

## **2. Geometry of the Decision Boundary**

The decision rule:

sign(w·x + b)

w is perpendicular (normal) to the decision boundary. The bias b shifts the line (in 2D) or hyperplane (in higher dimensions).

## Why is w perpendicular to the decision boundary?

In 2D, the decision boundary equation is:

w₀x₀ + w₁x₁ + b = 0

Think of this as:

w·x + b = 0

A dot product w·x is constant for all points on the boundary.

The set of all points with w·x = c is a line (or plane in higher dimensions) perpendicular to w.

That's because the dot product measures projection along w — so if you change your position along w, you change w·x; but if you move sideways (perpendicular to w), the dot product stays the same.

**Example in 2D:**

If w=(2,1), the boundary equation is:

2x + 1y + b = 0

The vector (2,1) points straight out from the line.

Walking parallel to the line keeps 2x + y the same.

Walking along (2,1) moves you "away" from the boundary.

📌 **Bias b**

If b = 0, the line passes through the origin.

Changing b shifts the line parallel to itself.

So b controls where the perpendicular line sits.

Example in 2D:

Equation w₀x₀ + w₁x₁ + b = 0 defines a line.

Points where w·x + b > 0 → classified as +1.

Points where w·x + b < 0 → classified as −1.

## 3. **The Update Rule**

Given a training point (xᵢ, yᵢ):

If correctly classified: do nothing.

If misclassified: update as:

w ← w + yᵢxᵢ
b ← b + yᵢ


### Why does this work?

If yᵢ = +1 and the point is misclassified, it means w·xᵢ + b < 0. Adding yᵢxᵢ moves w towards xᵢ, making w·xᵢ larger.

If yᵢ = −1 and it's misclassified, w·xᵢ + b > 0. Adding yᵢxᵢ moves w away from xᵢ, decreasing w·xᵢ.

## 4. **Convergence Guarantee**

If the data is linearly separable, PLA will converge in a finite number of steps.

Proof idea:

Show that the projection of w onto w* (the perfect separator) increases with every update.

Show that the length of w grows at most linearly with steps.

Use these two facts to bound the number of updates.

Bound:

**updates ≤ (R / γ)²** steps

where:

- R = max norm of any xᵢ
- γ = margin of separation

## **. Why does PLA converge?**

This is the **Convergence Theorem** for the Perceptron:

If the data is:

- Linearly separable
- Has maximum point length R
- Has a separation margin γ (distance from the closest point to the separating boundary)

Then **PLA will stop in at most**:

**updates ≤ (R / γ)²** steps.

---

### **Key intuition behind the proof**

Let:

w* = "perfect" weight vector that classifies all points correctly and has ||w*|| = 1.

γ = the margin, meaning: yᵢ(w* · xᵢ) ≥ γ for all i

R = maxᵢ ||xᵢ|| (largest point length).

---

### Step 1: Every update **aligns w more with w∗**

When you make a mistake on (xᵢ, yᵢ), you do:

w ← w + yᵢxᵢ

Since yᵢ(w* · xᵢ) ≥ γ, this update increases the projection:

w_new · w* ≥ w_old · w* + γ

So the "shadow" of w on w∗ grows by at least γ each mistake.

---

### Step 2: The length of w can't grow too fast

From the update:

||w_new||² = ||w_old||² + ||xᵢ||² + 2yᵢ(w_old · xᵢ)

The last term is **negative or zero** when we make a mistake,

so:

||w_new||² ≤ ||w_old||² + R²

After k mistakes:

||w_k|| ≤ R√k

---

### Step 3: Combine both bounds

From Step 1 (projection grows linearly):

w_k · w* ≥ kγ

From Step 2 (norm grows like √k):

w_k · w* ≤ ||w_k|| · ||w*|| ≤ R√k

So:

kγ ≤ R√k

Divide both sides by √k:

√k ≤ R/γ

Square both sides:

k ≤ (R/γ)²

✅ That's the formula.

---

### **The actual reason it converges**

The magic is in **both effects working together**:

- **Projection on w∗ keeps increasing** every mistake — you can't keep misclassifying forever without getting more aligned with the "right" direction.
- **Norm growth is bounded** — you can't have infinite increases in projection without the length catching up and violating the bound.

Eventually, these two constraints meet, forcing the algorithm to stop making mistakes.

## 5. **Limitations**

- **Only works** if the data is linearly separable (classic PLA).
- No notion of margin (unlike SVM).
- Sensitive to feature scaling — large-scale features dominate updates.
- May loop forever if data is not separable.

## 6. **Practical Tweaks**

- **Learning rate** η:
    
    Update rule becomes w ← w + ηyᵢxᵢ
    
    Doesn’t affect convergence for separable data but can help stability.
    
- **Shuffling data** each pass can help avoid cycles.
- **Normalizing features** makes learning more stable.

## 7. **Connections to Other Methods**

- PLA is **stochastic gradient descent** on the **perceptron loss**:

L(w,b) = ∑[i∈misclassified] −yᵢ(w·xᵢ + b)

- Related to **hinge loss** used in SVM, except SVM also enforces a margin.
- With non-linear features (via kernel trick) → becomes the **Kernel Perceptron**.