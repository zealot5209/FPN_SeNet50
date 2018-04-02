#__author__ = 'hujie'
# usage example: run "python draw_loss.py log_0-15000.txt 7000 log_7000-15000.txt 10000 log_10000-15000.txt "
# if you have only one log to plot, just run "python draw_loss.py log.txt"

import sys
from matplotlib import pyplot as plt

def parse_log(infile):
    f = open(infile,'r')
    train_iteration=[]
    train_loss=[]
    test_iteration=[]
    test_loss=[]
    test_accuracy=[]
    train_FPNClsLoss = []
    train_FPNLossBBox = []
    train_RcnnLossBBox = []
    train_RcnnLossCls = []
    for line in f.readlines():
        if 'Iteration' in line and 'loss' in line:
            split_line = line.strip().split()
            train_iteration.append(int(split_line[split_line.index('Iteration') + 1][:-1]))
            train_loss.append(float(split_line[split_line.index('loss') + 2]))
            continue
        if 'accuracy' in line and 'Test' in line:
            split_line = line.strip().split()
            test_accuracy.append(float(split_line[split_line.index('accuracy') + 2]))
            continue
        if 'Iteration' in line and 'Testing net' in line:
            split_line = line.strip().split()
            test_iteration.append(int(split_line[split_line.index('Iteration') + 1][:-1]))
            continue
        if 'loss' in line and 'Test' in line:
            split_line = line.strip().split()
            test_loss.append(float(split_line[split_line.index("loss") + 2]))
            continue

        if 'FPNClsLoss' in line and '=' in line:
            split_line = line.strip().split()
            train_FPNClsLoss.append(float(split_line[split_line.index("FPNClsLoss") + 2]))
            continue

        if 'FPNLossBBox' in line and '=' in line:
            split_line = line.strip().split()
            train_FPNLossBBox.append(float(split_line[split_line.index("FPNLossBBox") + 2]))
            continue

        if 'RcnnLossBBox' in line and '=' in line:
            split_line = line.strip().split()
            train_RcnnLossBBox.append(float(split_line[split_line.index("RcnnLossBBox") + 2]))
            continue

        if 'RcnnLossCls' in line and '=' in line:
            split_line = line.strip().split()
            train_RcnnLossCls.append(float(split_line[split_line.index("RcnnLossCls") + 2]))
            continue

    if len(test_iteration)!=len(test_accuracy):
        test_iteration.pop()
    if len(test_loss)!=len(test_accuracy):
        test_iteration.pop()
        test_accuracy.pop()
    return [train_iteration, train_loss, test_iteration, test_loss, test_accuracy, train_FPNClsLoss, train_FPNLossBBox, train_RcnnLossBBox, train_RcnnLossCls]


def draw_loss(arg):
  train_iteration, train_loss, test_iteration, test_loss, test_accuracy, train_FPNClsLoss, train_FPNLossBBox, train_RcnnLossBBox, train_RcnnLossCls = arg
  #fig = plt.figure(figsize=[25, 10])
  # fig1 = fig.add_subplot(121)
  # plt.plot(train_iteration,train_loss,'b',label = 'train')
  # plt.plot(test_iteration,test_loss,'r',label = 'test')
  # plt.title('Train and Test loss')
  # plt.xlabel('Iteration')
  # plt.ylabel('Loss')
  # plt.legend()
  #
  # fig2 = fig.add_subplot(122)
  # plt.plot(test_iteration,test_accuracy,'r')
  # plt.title('Test accuracy')
  # plt.xlabel('Iteration')
  # plt.ylabel('Accuracy')
  # plt.show()


  plt.figure(22)
  plt.subplot(221)
  plt.plot(train_iteration, train_FPNClsLoss[0:-1], 'r')
  plt.title('FPNClsLoss')
  plt.xlabel('Iteration')
  plt.ylabel('FPNClsLoss')

  plt.subplot(222)
  plt.plot(train_iteration, train_FPNLossBBox[0:-1], 'r')
  plt.title('FPNLossBBox')
  plt.xlabel('Iteration')
  plt.ylabel('FPNLossBBox')

  plt.subplot(223)
  plt.plot(train_iteration, train_RcnnLossBBox[0:-1], 'r')
  plt.title('RcnnLossBBox')
  plt.xlabel('Iteration')
  plt.ylabel('RcnnLossBBox')

  plt.subplot(224)
  plt.plot(train_iteration, train_RcnnLossCls[0:-1], 'r')
  plt.title('RcnnLossCls')
  plt.xlabel('Iteration')
  plt.ylabel('RcnnLossCls')
  plt.show()

def draw_loss_from_file(infile):
  arg = parse_log(infile)
  draw_loss(arg)


def concat_loss(file_list, iter_list):
    concat_result = [[],[],[],[],[],[],[],[],[]]
    index_test_flag = False
    if len(file_list)!=len(iter_list):
        iter_list.append(parse_log(file_list[-1])[0][-1])
    for i,file in enumerate(file_list):
        result = parse_log(file)
        index_train = result[0].index(iter_list[i])
        if iter_list[i] not in result[2]:
          index_test_flag = True
        else:
          index_test = result[2].index(iter_list[i])
        result[0] = result[0][:index_train]
        result[1] = result[1][:index_train]
        if index_test_flag == False:
          result[2] = result[2][:index_test]
          result[3] = result[3][:index_test]
          result[4] = result[4][:index_test]
        for j in xrange(9):
            concat_result[j].extend(result[j])
    return concat_result;



if __name__ == '__main__':
    file_list=[]
    iter_list=[]
    for i,arg in enumerate(sys.argv):
        if i==0: continue
        if i%2 == 0:
            iter_list.append(int(arg))
        else:
            file_list.append(arg)
    result = concat_loss(file_list,iter_list)
    draw_loss(result)

