from mnistdnn import mnistnet


model   = mnistnet(minibatchsize=32, learningrate = 0.01)


model.buildnet()

model.loaddata()

model.init_net()

model.data_mode(1)
model.train_mode(1)

epoch=0


acc = []
loss =[]

for ii in range(100000):    
    model.next_batch()


    model.train_net( )
    
    if epoch < model.epoch:
        epoch = model.epoch
        model.save_model('exp1')
    

    if ii % 1000 == 0 :
        model.fill_train_data()
        model.calloss()
        loss.append(model.v_loss)
        
        model.calacc()
        acc.append(model.v_acc)
        print('epoch',model.epoch,'loss', model.v_loss,'acc',model.v_acc,'lr',model.lr)

