# Saliency Plots

These plots explain which input features the trained DQN agent pays attention to when making driving decisions. Each group of plots is generated for 3 sampled observations (step00, step01, step02) using 4 attribution methods: Vanilla Gradient, Gradient x Input (GxI), SmoothGrad, and Integrated Gradients (IG).

| File | Description |
|------|-------------|
| `07_saliency_bar_<method>_step<N>.png` | Bar chart of the top 15 most influential input features for a given observation, ranked by attribution magnitude under the specified method. |
| `08_saliency_heatmap_<method>_step<N>.png` | Heatmap laying out attribution scores across all 25 input features (5 vehicles x 5 features each), making it easy to see which vehicles and which feature types drive the agent's decision. |
| `09_saliency_methods_comparison_step<N>.png` | Side-by-side 2x2 grid comparing all four attribution methods on the same observation, highlighting where they agree and where they differ. |
| `10_ig_completeness_step<N>.png` | Verification plot for Integrated Gradients showing that the sum of all attributions equals the difference in Q-value between the input and baseline, confirming mathematical correctness (error below 0.02%). |
| `11_avg_attribution_summary.png` | Averaged absolute attribution across all sampled observations, showing which features are consistently important regardless of the specific driving scenario. |
