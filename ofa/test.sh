device=7

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA datase5
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python main.py --dataset mvtec \
        --data_path /home/lwx/mvtec --save_path ./results/${base_dir}/ \
        --checkpoint_path ${save_dir}epoch_15.pth \
         --features_list 6 12 18 24 --image_size 224 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA datase5
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python main.py --dataset visa \
#         --data_path /home/lwx/visa --save_path ./results/${base_dir}/ \
#         --checkpoint_path ${save_dir}epoch_15.pth \
#          --features_list 6 12 18 24 --image_size 224 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
#     wait
#     done
# done

