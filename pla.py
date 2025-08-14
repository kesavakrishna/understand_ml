import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------------
# STEP 1: Generate a slightly harder separable dataset
# --------------------------
np.random.seed(42)  # For reproducible results (same random numbers each run)

# Generate class +1 points around the mean (1, 1)
X_pos = np.random.randn(20, 2) + np.array([1, 1]) #creates 20 x 2 matrix of samples from standard normal distribution(mean 0, standard deviation 1) and shifts all points by (1,1) so they are centered around 1,1 instead of 0,0
y_pos = np.ones(len(X_pos))  # creates a vector of n ones(20 here) and assigns +1 to all the points

# Generate class -1 points around the mean (-1, -1)
X_neg = np.random.randn(20, 2) + np.array([-1, -1]) #same as above, but is now centered around -1,-1
y_neg = -np.ones(len(X_neg))  # same as above, but assigns -1 to all the points

# Stack the two classes into one dataset
X = np.vstack((X_pos, X_neg))  # Shape: (40, 2)
y = np.hstack((y_pos, y_neg))  # Shape: (40,)

# --------------------------
# STEP 2: Find a "perfect" separating vector w* (for reference)
# --------------------------
# This is NOT part of PLA — we just compute it so we can show it in the animation
w_star, b_star = np.zeros(2), 0  # Start from zero vector of length 2 and zero bias
for _ in range(50):  # Limit to 50 passes max
    errors = 0
    for xi, yi in zip(X, y): #iterates over points x and y together
        # If the current w and b misclassify a point:
        if yi * (np.dot(w_star, xi) + b_star) <= 0: #if this formula is true, then its misclassified
            w_star += yi * xi  # Move w in the direction of the misclassified point
            b_star += yi       # Adjust bias accordingly
            errors += 1
    if errors == 0:
        break  # Stop when all points are correctly classified

# --------------------------
# STEP 3: Perceptron Algorithm with step history
# --------------------------
def perceptron_with_history(X, y, max_updates=100):
    """
    Runs PLA and records w, b after each update, but stops at max_updates.
    """
    w = np.zeros(X.shape[1])  # Start with w = (0, 0)
    b = 0                     # Start with bias = 0
    history = []              # Store (w, b) at each update
    updates = 0               # Count number of updates made
    
    while updates < max_updates:
        errors_this_round = 0
        for xi, yi in zip(X, y):
            # Check if current point is misclassified
            if yi * (np.dot(w, xi) + b) <= 0:
                # Update rule: w <- w + y*xi, b <- b + y
                w += yi * xi
                b += yi
                updates += 1
                history.append((w.copy(), b))  # Save current weights and bias. We are copying the weights because they are stored as references(lists and arrays are stored as references). so if we save w directly into history, the next time w changes, all the previous entries will also change
                errors_this_round += 1
                
                # If we hit the maximum updates, stop immediately
                if updates >= max_updates:
                    return history
        # If no misclassifications in a full pass, stop early
        if errors_this_round == 0:
            break
    return history

# Run PLA and store history
history = perceptron_with_history(X, y, max_updates=100)

# --------------------------
# STEP 4: Animation setup
# --------------------------
fig, ax = plt.subplots(figsize=(6,6))

def animate(i):
    """
    This function draws one frame of the animation.
    """
    ax.clear()  # Clear previous frame
    w, b = history[i]  # Get weights and bias at this step
    
    # Plot the two classes of points
    ax.scatter(X_pos[:,0], X_pos[:,1], color='blue', label='+1')
    ax.scatter(X_neg[:,0], X_neg[:,1], color='red', label='-1')
    
    # Draw the decision boundary (only if w[1] != 0 to avoid divide-by-zero)
    if w[1] != 0:
        x_vals = np.linspace(-4, 6, 100)                  # X range for line
        y_vals = -(w[0]*x_vals + b) / w[1]                # Line equation
        ax.plot(x_vals, y_vals, 'k--', label='Decision boundary')
    
    # Draw the current w vector (green arrow)
    ax.arrow(0, 0, w[0], w[1],
             head_width=0.2, head_length=0.3,
             fc='green', ec='green', linewidth=2, label='Current w')
    
    # Draw the perfect w* vector (red arrow) for reference
    ax.arrow(0, 0, w_star[0], w_star[1],
             head_width=0.2, head_length=0.3,
             fc='red', ec='red', linewidth=2, label='Perfect w*')
    
    # Formatting the plot
    ax.set_xlim(-4, 6)
    ax.set_ylim(-4, 6)
    ax.grid(True)
    ax.set_title(f"PLA - Step {i+1}/{len(history)}")
    ax.legend()

# Create animation (interval = ms between frames)
ani = FuncAnimation(fig, animate, frames=len(history), interval=200, repeat=False)

# Show animation
plt.show()
