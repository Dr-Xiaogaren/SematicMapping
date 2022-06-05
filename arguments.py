import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Goal-Oriented-Semantic-Exploration')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=1)
    parser.add_argument('--total_num_scenes', type=str, default="1")
    parser.add_argument('-n', '--num_processes', type=int, default=1,
                        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""")
    parser.add_argument('--num_processes_per_gpu', type=int, default=1)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=1)
    parser.add_argument('--eval', type=int, default=0,
                        help='0: Train, 1: Evaluate (default: 0)')
    parser.add_argument('--num_training_frames', type=int, default=10000000,
                        help='total number of training frames')
    parser.add_argument('--num_eval_episodes', type=int, default=200,
                        help="number of test episodes per scene")
    parser.add_argument('--num_train_episodes', type=int, default=10000,
                        help="""number of train episodes per scene
                                before loading the next scene""")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=-1,
                        help="""gpu id for semantic model,
                                -1: same as sim gpu, -2: cpu""")
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                       the predicted semantic map,
                                    2: Render the observation with semantic
                                       predictions and the predicted semantic map
                                    (default: 0)""")


    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')
    parser.add_argument('--num_sem_categories', type=float, default=16)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9,
                        help="Semantic prediction confidence threshold")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=2)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    parser.add_argument('--global_num_step', type=int, default=20, help="Number of local steps in a global step")
    parser.add_argument('--random_initial_location', type=bool, default=True)
    parser.add_argument('--min_initial_distance', type=float, default=0.5,
                        help="Minimum distance among random initial robot location")
    parser.add_argument('--max_initial_distance', type=float, default=5.0,
                        help="Maximum distance among random initial robot location")

    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            args.total_num_scenes = int(args.total_num_scenes)
            # GPU Memory required for the SemExp model:
            #       0.8 + 0.4 * args.total_num_scenes (GB)
            # GPU Memory required per thread: 2.6 (GB)
            min_memory_required = max(0.8 + 0.4 * args.total_num_scenes, 2.6)
            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                                 torch.cuda.get_device_properties(
                                     i).total_memory
                                 / 1024 / 1024 / 1024)
                assert gpu_memory > min_memory_required, \
                    """Insufficient GPU memory for GPU {}, gpu memory ({}GB)
                    needs to be greater than {}GB""".format(
                        i, gpu_memory, min_memory_required)

            num_processes_per_gpu = int(gpu_memory / 2.6)
            num_processes_on_first_gpu = \
                int((gpu_memory - min_memory_required) / 2.6)

            if args.eval:
                max_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                assert max_threads >= args.total_num_scenes, \
                    """Insufficient GPU memory for evaluation"""

            if num_gpus == 1:
                args.num_processes_on_first_gpu = 1
                args.num_processes_per_gpu = 0
                args.num_processes = 1
                assert args.num_processes > 0, "Insufficient GPU memory"
            else:
                num_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                num_threads = min(num_threads, args.total_num_scenes)
                args.num_processes_per_gpu = num_processes_per_gpu
                args.num_processes_on_first_gpu = max(
                    0,
                    num_threads - args.num_processes_per_gpu * (num_gpus - 1))
                args.num_processes = num_threads

            args.sim_gpu_id = 1

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print("Number of processes on GPU 0: {}".format(
                args.num_processes_on_first_gpu))
            print("Number of processes per GPU: {}".format(
                args.num_processes_per_gpu))
    else:
        args.sem_gpu_id = -2


    return args
