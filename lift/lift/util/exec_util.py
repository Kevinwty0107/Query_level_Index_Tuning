import subprocess


def local_exec(cmd):
    """
    Executes command via subprocess and prints results.

    Args:
        cmd (str): Shell command.
    """
    print("Executing local exec = " + cmd)
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True)
    for line in p.stdout:
        print(line)
    output, error = p.communicate()
    if p.returncode != 0:
        for line in error.decode().split('\n'):
            print(line)
        for line in output.decode().split('\n'):
            print(line)

    p.wait()
