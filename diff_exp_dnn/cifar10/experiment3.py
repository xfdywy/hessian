from cifar10dnn import mnistnet
import numpy as np
import matplotlib.pyplot as plt
import pickle
model   = mnistnet(minibatchsize=50, learningrate = 0.1)


model.buildnet()

model.loaddata()

model.init_net()

model.data_mode(1)
model.train_mode(1)

epoch=0

train_acc = []
train_loss =[]
test_acc = []
test_loss =[]

weight = []

train_grad_norm_l2 = []
train_grad_norm_max = []
train_grad_norm_l1 = []


test_grad_norm_l2 = []
test_grad_norm_max = []
test_grad_norm_l1 = []

dis =[]
#model.lr = 0.01


temp_step = 0


file_name =  '_'.join(model.info.values())


printoutfile = open(file_name + '_printout.txt','w')

for ii in range(10000000): 
    
#        if model.epoch_final  ==  False:
#            break
#    print(ii)

    if model.epoch >40:
        break
    
    model.global_step = 0
    model.next_batch()   
    model.train_net( )
    
    if model.epoch_final == True:
        model.eval_weight()
        weight.append(model.v_weight)
    

    

#            model.save_model('exp1')
    
    if model.data_point % (model.one_epoch_iter_num // 5 ) == 0 :

        model.fill_train_data()
        model.calloss()
        train_loss.append(model.v_loss)            
        model.calacc()
        train_acc.append(model.v_acc)
        model.eval_grad()
        
        train_grad_norm_l2.append(model.v_grad_norm_l2)
        train_grad_norm_max.append(model.v_grad_norm_max)
        train_grad_norm_l1.append(model.v_grad_norm_l1)
        
        
        model.fill_test_data()
        model.calloss()
        test_loss.append(model.v_loss)            
        model.calacc()
        test_acc.append(model.v_acc)
        model.eval_grad()
        
        test_grad_norm_l2.append(model.v_grad_norm_l2)
        test_grad_norm_max.append(model.v_grad_norm_max)
        test_grad_norm_l1.append(model.v_grad_norm_l1)     
        
        
        
        print("##epoch:%d##  loss : %f/%f ,   acc : %f/%f ,   lr : %f" 
              % (model.epoch,train_loss[-1] , test_loss[-1] ,  train_acc[-1],test_acc[-1],  model.lr))
        print("##epoch:%d##  loss : %f/%f ,   acc : %f/%f ,   lr : %f" 
              % (model.epoch,train_loss[-1] , test_loss[-1] ,  train_acc[-1],test_acc[-1],  model.lr) , file = printoutfile)
        
        
        temp_step += 1
        

        if  temp_step >5 and  np.mean(train_loss[-5:]) -train_loss[-1]  < (model.lr/100.0 ):
            model.lr = model.lr / 10.0
            temp_step = 0
            print('learning rate decrease to ', model.lr)
            print('learning rate decrease to ', model.lr,file = printoutfile)
 
all_res = {'info' : 'dnn_cifar10_sgd' ,
           'train_acc' :train_acc,
            'train_loss':train_loss,
            'test_acc' :test_acc,
            'test_loss' :test_loss,
            
            'weight' :weight,
            
            'train_grad_norm_l2' :train_grad_norm_l2,
            'train_grad_norm_max' :train_grad_norm_max,
            'train_grad_norm_l1': train_grad_norm_l1,
            
            
            'test_grad_norm_l2' :test_grad_norm_l2,
            'test_grad_norm_max' :test_grad_norm_max,
            'test_grad_norm_l1' :test_grad_norm_l1
           }         


with open('./all_res/dnn_cifar10_sgd.plk','wb') as f  :
    pickle.dump(all_res , f)
model.save_model('../modelsave/' ,'dnn_cifar10_sgd')        
#    model.fill_train_data()
#    model.eval_grad()
#    print(model.v_grad_max)
#    print(model.v_grad_min)
#    print(model.v_grad_norm)
#    grad_norm.append(model.v_grad_norm)
##    
#    model.eval_weight()
#    weight.append(model.v_weight)
#    
#    dis_1 = np.linalg.norm(weight[-1]-weight[-2])
#    dis.append(dis_1)   
#    print(dis_1 )
