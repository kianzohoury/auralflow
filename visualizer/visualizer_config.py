import os
import subprocess
import sys
import threading

#
# def run_tensorboard(log_dir):
#     # os.chdir(Path(__file__).parent / "runs")
#     # [f"python -m tensorboard.main --logdir {log_dir}"]
#
#     command = ["tensorboard", "--logdir", "logs"]
#     # command = ["python", "-m", "tensorboard.main", "--logdir", log_dir]
#     # command = tb_path
#     # print(command)
#     # tb_main.main()
#     # log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
#     # Start tensorboard server
#     # tb.program.FLAGS.logdir = os.getcwd() + '/logs'
#     # tb.program.main(tb.default.get_plugins(),
#     #             tb.default.get_assets_zip_provider())
#     # tb_ = program.TensorBoard(tb_def.get_plugins())
#     # tb.configure(argv=[None, '--logdir', 'logs'])
#     # url = tb.launch()
#     # sys.stdout.write('TensorBoard at %s \n' % url)
#     # print(os.system("tensorboard --logdir runs"))
#
#     tb_thread = threading.Thread(
#         target=lambda: subprocess.Popen(
#             command, shell=False, cwd=os.getcwd()
#         ).communicate(),
#         daemon=True,
#     )
#     tb_thread.start()
#     # print(subprocess.Popen(command, shell=False, cwd=os.getcwd()).communicate())
#     # print(subprocess.check_output(command), shell=True)
#     sys.exit(1)
