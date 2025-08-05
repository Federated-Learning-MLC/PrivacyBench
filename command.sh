privacybench --help                    # ✅ Your proper CLI working!
privacybench list experiments          # ✅ Show experiment table
privacybench run --experiment cnn_baseline --dataset alzheimer --dry-run  # ✅ Dry run

# Phase 3: Individual config files (NEW)
privacybench run --config cnn_alzheimer
privacybench run --config configs/experiments/privacy/dp_configurations.yaml
privacybench validate --config fl_dp_cnn_alzheimer

# Phase 1: Legacy CLI args (PRESERVED)
privacybench run --experiment cnn_baseline --dataset alzheimer
privacybench run --experiment fl_dp_cnn --dataset skin_lesions --dry-run

# Enhanced listing
privacybench list configs
privacybench list all