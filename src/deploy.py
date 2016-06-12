#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Quick & dirty deployment script
'''

import logging
import os
import sys
import paramiko
import getpass
from scp import SCPClient
from util import helper

_logger = logging.getLogger(__name__)


def setup_logging(log_level):
    logging.basicConfig(
        filename=None,
        level=log_level,
        format='%(asctime)s: %(levelname)7s: [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def deploy(ssh_client, config):
    files_to_be_uploaded = []
    dirs_to_be_uploaded = config["dirs_to_be_uploaded"]

    for directory in dirs_to_be_uploaded:
        for subdir, _, files in os.walk(directory):
            subdir_parts = subdir.split(os.path.sep)
            if len(filter(lambda part: part not in config["exclude_folders"], subdir_parts)) < len(subdir_parts):
                continue # exclude

            filtered_files = filter(lambda f: len(f.split(".")) == 1 or ('.'+f.split(".")[len(f.split(".")) - 1]) not in config["exclude_file_ext"], files)
            files_to_be_uploaded += map(lambda f: (subdir.replace(helper.APP_PATH, ""), os.path.join(subdir, f)), filtered_files)

    n_files = len(files_to_be_uploaded)
    _logger.info("Total number of files to transfer: %d", n_files)

    scp = SCPClient(ssh_client.get_transport())
    progress_bar = helper.ProgressBar(n_files)
    already_created_dirs = set()
    for (relative_dir_path, local_file_path) in files_to_be_uploaded:
        relative_file_path = local_file_path.replace(helper.APP_PATH, "")
        upload_file = True
        helper
        md5_hash = helper.compute_hash_of_file(local_file_path)
        try:
            if relative_dir_path not in already_created_dirs:
                _logger.info("Directory: %s", relative_dir_path)
                ssh_client.exec_command('mkdir -p ' + './MLSE' + relative_dir_path)
                already_created_dirs.add(relative_dir_path)

            _, stdout, _ = ssh_client.exec_command('md5sum ' + './MLSE' + relative_file_path)
            line = stdout.next()
            if line.split()[0].strip() == md5_hash:
                upload_file = False
        except:
            pass
        if upload_file:
            _logger.debug("Transfering file: %s", local_file_path.replace(helper.APP_PATH, ""))
            scp.put(local_file_path, './MLSE' + relative_dir_path, recursive=False)
        progress_bar.update()
    _logger.info("Finished!")
    progress_bar.finish()
    return True


if __name__ == "__main__":
    if len(sys.argv) < 4:
        helper.error("usage: python deploy.py <hostname_or_ip> <port> <username> [--not-deploy]")

    config = {
        "dirs_to_be_uploaded": [helper.APP_PATH],
        "exclude_folders": [
            ".settings",
            "cache",
            "docs",
            "presentation",

            # directory of local git repository
            ".git",

            # local datasets
            #"programmers.stackexchange.com",
            #"quarter",
            #"academia",
            #"android.stackexchange.com",
            #"apple.stackexchange.com",
            "codereview.stackexchange.com"
        ],
        "exclude_file_ext": [".gitignore", ".7z", ".pyc", ".project", ".pydevproject", ".DS_Store"]
    }

    setup_logging(logging.INFO)
    password = None
    while password is None or not isinstance(password, (str, unicode)) or len(password) <= 2:
        password = getpass.getpass("Please enter password: ")

    ssh_client = None
    ssh_client = create_ssh_client(sys.argv[1], int(sys.argv[2]), sys.argv[3], password)
    if ssh_client is None:
        helper.error("Deployment failed!")

    if len(sys.argv) <= 4 or sys.argv[4].lower() != "--not-deploy":
        result = deploy(ssh_client, config)
        if not result:
            helper.error("Deployment failed!")
        print "Deployment finished!"

    # execute job on server
    data_set = "quarter"
    # #_, stdout, stderr = ssh_client.exec_command('cd MLSE/src/ && python -m main ../data/example')
    # _, stdout, stderr = ssh_client.exec_command('cd MLSE/src/ && screen -dmS MLSE.%s python -m main ../data/%s' % (data_set, data_set))
    # for line in stdout:
    #     if line is not None and len(line) > 0:
    #         print line
    # for line in stderr:
    #     if line is not None and len(line) > 0:
    #         print line
    # print "Started job in new screen"
