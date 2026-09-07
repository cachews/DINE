
for seed in 1 2 3 4 5; do
    python -u run.py \
        --name DINE-example \
        --mlflow_experiment $mlflow_experiment \
        --seed $seed \
        --method dual \
        --model DINE \
        --data M5 \
        --seq_len 28 \
        --pred_len 1 \
        --test_points 300 \
        --vali_points 300 \
        --trim_leading_zeros \
        --feature_separating_periods \
        --feature_successive_zero_periods \
        --filter_intermittent_lumpy \
        --batch_size 64 \
        --train_epochs 300 \
        --patience 50 \
        --loss MSE \
        --bce_use_weight \
        --dual_demand_loss_weight \
        --dual_demand_loss_weight_multiplier 0.5 \
        --pooling last \
        --size_branch_residual
done