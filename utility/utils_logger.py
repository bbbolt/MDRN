import logging
import os

import numpy


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(message)s')
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


def del_after_break_point_info(log_path: str, epoch_checkpoint, checkpoint_internal):
    with open(log_path) as f:
        a = f.readlines()
        need_str = '==Epoch {}=='.format(epoch_checkpoint)
        save_str = '==Epoch {}=='.format(epoch_checkpoint + checkpoint_internal - 1)
        count = 1
        new = []
        for i in a:
            if need_str in i:
                r_idx = a.index(i)
                new = a[:r_idx]
                count -= 1
                if count < 0:
                    raise KeyError('多个指定字符冲突，请检查是否包含多个指定字符串')
            if save_str in a:
                raise KeyError('请检查是否未重置断点数值next_break_point')
        if new:
            with open(log_path, 'w') as g:
                g.writelines(new)
        else:
            pass


def get_earliest_checkpoint(check_path, epoch=True):
    relpath = os.path.relpath(check_path)
    get_dir_list = []
    for i in os.listdir(relpath):
        if 'ipynb' not in i:
            get_dir_list.append(i)
    empty = []
    for name in get_dir_list:
        time = os.path.getmtime(os.path.join(relpath, name))
        empty.append(time)
    if not empty:
        return False, False

    empty = numpy.array(empty)
    idx = empty.argsort().tolist()[::-1][0]
    check_name, _ = os.path.splitext(get_dir_list[idx])
    if epoch:
        try:
            return os.path.join(relpath, get_dir_list[idx]), int(check_name.split(sep='epoch_')[1])
        except ValueError:
            print('请检查是否为epoch记录！')
            raise ValueError
    else:
        try:
            return os.path.join(relpath, get_dir_list[idx]), int(check_name.split(sep='step')[1])
        except ValueError:
            print('请检查是否为step记录！')
            raise ValueError


def get_earliest_checkpoint_best_psnr(check_best_path):
    relpath = os.path.relpath(check_best_path)
    get_dir_list = os.listdir(relpath)
    empty = []
    for name in get_dir_list:
        if 'ipynb' not in name:
            time = os.path.getmtime(os.path.join(relpath, name))
            empty.append(time)
    if not empty:
        return 0

    empty = numpy.array(empty)
    idx = empty.argsort().tolist()[::-1][0]
    check_name, _ = os.path.splitext(get_dir_list[idx])

    return float(check_name.split(sep='PSNR_Y_')[1])


if __name__ == '__main__':
    print(get_earliest_checkpoint('F:\CNN_Trans\checkpoint'))
    print(get_earliest_checkpoint_best_psnr('F:\CNN_Trans\checkpoint_best'))
