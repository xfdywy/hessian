from mnistdnn import mnistnet
import numpy as np

model   = mnistnet(minibatchsize=64, learningrate = 0.01)


model.buildnet()

model.loaddata()

model.init_net()

model.data_mode(1)
model.train_mode(2)

epoch=0


acc = []
loss =[]


weight = []

grad_norm = []

dis =[]


for jj in range(30):
    model.eval_weight()
    weight.append(model.v_weight)
    for ii in range(10000): 
        model.lr = 0.01
        model.global_step = 0
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
            
        
    model.fill_train_data()
    model.eval_grad()
    print(model.v_grad_max)
    print(model.v_grad_min)
    print(model.v_grad_norm)
    grad_norm.append(model.v_grad_norm)
    
    model.eval_weight()
    weight.append(model.v_weight)
    
    dis_1 = np.linalg.norm(weight[-1]-weight[-2])
    dis.append(dis_1)   
    print(dis_1 )
