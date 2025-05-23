python3 predict.py --point-cloud ../test_data/forest.ply \
    --model global.pth \
    --batch_size 4 \
    --is-wood 0.50 \
    --grid_size 2.0 4.0 \
    --min_pts 128 \
    --max_pts 16384 \
    --preserve_raycloud_format \
    --verbose;