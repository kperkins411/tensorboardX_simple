'''
Demos simple tensorboardX, you can do a lot with this see
https://github.com/lanpa/tensorboardX

when its run will create a runs directory
to see this projects results in tensorboard go to terminal and type

tensorboard --logdir=~/fastai2/fastai/courses/dl2/tensorboardX_simple
(or wherever the app is installed)

then open browser to:
http://127.0.0.1:6006
(or whatever address the above command gives)
'''


from tensorboardX import SummaryWriter
from numpy.random import rand

# tensorboard tracker
writer = SummaryWriter()
NUMB_EPOCHS = 100

trn_lss, trn_acc = 100.0,0.0
tst_lss, tst_acc = 10.0,0.0
for epoch in range(NUMB_EPOCHS):  # loop over the dataset multiple times
    #generate some metrics to track
    trn_lss = trn_lss - trn_lss/2.0
    tst_lss = trn_lss +1
    trn_acc = epoch/(1+epoch)
    tst_acc = trn_acc- 0.01*rand()

    #write em all out
    writer.add_scalars('losses', {"trn_lss": trn_lss,
                                  "tst_lss": tst_lss}, epoch)

    writer.add_scalars('accuracy', {"trn_acc": trn_acc,
                                  "tst_acc": tst_acc}, epoch)

    #some demo rubbish
    writer.add_scalar('twse/0050', rand(), epoch)
    writer.add_scalar('twse/2330', rand(), epoch)

print('Finished Training')
writer.close()
