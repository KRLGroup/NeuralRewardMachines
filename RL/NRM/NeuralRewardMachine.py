import os
import torchvision
from PIL import Image
import torch
import pickle
from .DeepAutoma import ProbabilisticAutoma
from .NN_models import CNN_grounder, Linear_grounder

from statistics import mean
from sklearn.model_selection import train_test_split

from .utils import eval_acceptance, eval_learnt_DFA_acceptance, eval_image_classification_from_traces
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import time

def create_batches_same_length(dataset, labels, size):
    new_dataset = []
    new_labels = []
    num_batches = int(len(dataset)/size)
    for i in range(num_batches):
        batch_trace = []
        batch_label = []
        for j in range(size):
            batch_trace.append(dataset[i*size+j])
            batch_label.append(labels[i*size+j])
        batch_trace = torch.stack(batch_trace)
        batch_label = torch.stack(batch_label)
        new_dataset.append(batch_trace)
        new_labels.append(batch_label)
    return new_dataset, new_labels 

class NeuralRewardMachine:
    def __init__(self, numb_states, numb_symbols, numb_rewards, num_exp=0,log_dir="Results/", dataset="minecraft_location"):
        self.first_training = False
        self.ltl_formula_string = "goal"
        self.log_dir = log_dir
        self.exp_num=num_exp

        self.numb_of_symbols = numb_symbols
        self.numb_of_states = numb_states
        self.numb_of_rewards = numb_rewards

        self.alphabet = ["c"+str(i) for i in range(self.numb_of_symbols) ]

        #################### networks
        self.hidden_dim =numb_states

        ##### DeepDFA
        self.deepAutoma = ProbabilisticAutoma(self.numb_of_symbols, self.numb_of_states, self.numb_of_rewards)

        ##### Classifier
        self.dataset = dataset
        if dataset == 'minecraft_image':
            self.num_classes = 5
            self.num_channels = 3

            self.pixels_h = 64
            self.pixels_v = 64

            self.num_features = 4 #<---??
            self.classifier = CNN_grounder(self.num_classes)

        if dataset == 'minecraft_location':
            self.num_inputs = 2
            self.num_classes = 5

            self.classifier = Linear_grounder(self.num_inputs, 8, self.num_classes)

        self.temperature = 1.0
        #questa resize si può togliere mi sà
        resize = torchvision.transforms.Resize((64,64))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            resize,
        ])

        #queste cose sotto misà che servivano per fare la image classification
        '''
        if dataset == 'minecraft_image':
            trace = []
            dir = os.listdir('custom_trace_whole')
            for i in range(len(dir)):
                img = Image.open('custom_trace_whole/img'+str(i)+'.jpg')
                img = transforms(img)
                trace.append(img)
            self.custom_trace = [torch.stack(trace).unsqueeze(0)]

            trace = []
        '''
        if dataset == 'minecraft_location':
            self.custom_trace = [torch.tensor([[0,0],[0,1],[0,2],[0,3],
                                             [1,0],[1,1],[1,2],[1,3],
                                             [2,0],[2,1],[2,2],[2,3],
                                             [3,0],[3,1],[3,2],[3,3]]).unsqueeze(0)]

        self.symbolic_grid = [torch.tensor([[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,1,0,0],
                                    [0,0,0,0,1], [1,0,0,0,0], [0,0,0,0,1], [0,0,0,0,1],
                                    [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1],
                                    [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,1], [0,1,0,0,0]])]
        #[0,0,0,0,1] white cell
        #[0,0,0,1,0] gem
        #[0,0,1,0,0] door
        #[0,1,0,0,0] lava
        #[1,0,0,0,0] pick

    # def set_dataset(self, image_traj, rew_traj):

    #     # print(len(rew_traj[0]))

    #     dataset_traces = []
    #     dataset_acceptances = torch.FloatTensor(rew_traj)
    #     for i in range(len(image_traj)):
    #         trace = []
    #         for img in image_traj[i]:
    #             trace.append(img)
    #         trace_tensor = torch.stack(trace)
    #         trace_tensor = torch.squeeze(trace_tensor)
    #         dataset_traces.append(trace_tensor)

    #     #train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = train_test_split(dataset_traces, dataset_acceptances, train_size=1, shuffle=True)
    #     train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = (dataset_traces, dataset_traces, dataset_acceptances, dataset_acceptances)

    #     train_img_seq, train_acceptance_img = create_batches_same_length(train_traces, train_acceptance_tr, 40)


    #     test_img_seq_hard, test_acceptance_img_hard = train_img_seq, train_acceptance_img

    #     image_seq_dataset = (train_img_seq, [], train_acceptance_img, test_img_seq_hard, [], test_acceptance_img_hard)
    #     self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset
    #     return

    def set_dataset(self, image_traj, rew_traj):
        
        dataset_acceptances = torch.LongTensor(rew_traj)
        dataset_traces = torch.stack([torch.stack(inner) for inner in image_traj])

        image_seq_dataset = ([dataset_traces], [], [dataset_acceptances], [dataset_traces], [], [dataset_acceptances])
        self.train_img_seq, self.train_traces, self.train_acceptance_img, self.test_img_seq_hard, self.test_traces, self.test_acceptance_img_hard = image_seq_dataset

        return image_seq_dataset

    def eval_learnt_DFA(self, automa_implementation, temp, mode="dev"):
        if mode=="dev":
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.dev_traces, self.dev_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr), automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.dev_traces, self.dev_acceptance_tr), automa_implementation, temp)
        else:
            if automa_implementation == 'dfa':
                train_acc = eval_learnt_DFA_acceptance(self.dfa, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp, alphabet=self.alphabet)
                test_acc = eval_learnt_DFA_acceptance(self.dfa, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp, alphabet=self.alphabet)
            else:
                train_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.train_traces, self.train_acceptance_tr),
                                                       automa_implementation, temp)
                test_acc = eval_learnt_DFA_acceptance(self.deepAutoma, (self.test_traces, self.test_acceptance_tr),
                                                      automa_implementation, temp)
        return train_acc, test_acc


    def train_symbol_grounding(self, num_of_epochs):

        #print(self.deepAutoma.trans_prob)
        #print(self.deepAutoma.rew_matrix)
        #assert False

        tot_size = len(self.train_img_seq)

        self.deepAutoma.to(device)
        '''
        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_train_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_train_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_test_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_test_acc_NS_exp"+str(self.exp_num), 'w')
        '''
        self.classifier.to(device)

        cross_entr = torch.nn.CrossEntropyLoss()
        print("_____________training the GROUNDER_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = self.classifier.parameters()
        if self.dataset == "minecraft_location":
            optimizer = torch.optim.Adam(params, lr=0.005)#, weight_decay=1e-3)
        else:
            optimizer = torch.optim.Adam(params, lr=0.01)#, weight_decay=1e-3)

        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-05)

        epoch = 0
        mean_loss = 1000000
        max_accuracy = 0
        best_classifier = self.classifier

        for _ in range(num_of_epochs):
        #while True:
            #if epoch % 40 == 0:
            print("epoch: ", epoch)
            epoch+=1
            losses = []

            for b in range(len(self.train_img_seq)):

                batch_img_seq = self.train_img_seq[b].to(device)

                batch_size = batch_img_seq.size()[0]
                length_seq = batch_img_seq.size()[1]
                target_rew_seq = self.train_acceptance_img[b].type(torch.int64).to(device)

                optimizer.zero_grad()
                if self.dataset == 'minecraft_image':
                    sym_sequences = self.classifier(batch_img_seq.view(-1, self.num_channels, self.pixels_v , self.pixels_h))
                else:
                    sym_sequences = self.classifier(batch_img_seq.double())

                sym_sequences = sym_sequences.view(batch_size, length_seq, self.numb_of_symbols)

                pred_states, pred_rew = self.deepAutoma(sym_sequences, self.temperature)

                pred_rew = pred_rew.view(-1, self.numb_of_rewards).to(device)
                target_rew = target_rew_seq.view(-1)
                loss = cross_entr(pred_rew, target_rew)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            
            
            mean_loss_new = mean(losses)
            sheduler.step(mean_loss_new)

            if mean_loss_new < 0.3 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > num_of_epochs/2 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new

            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_all(automa_implementation='logic_circuit', temperature=1, discretize_labels=True)
            if train_accuracy>= max_accuracy:
                max_accuracy = train_accuracy
                best_classifier = self.classifier

            if self.dataset == "minecraft_location":
                train_image_classification_accuracy, test_image_classification_accuracy = self.eval_image_classification()
            else:
                train_image_classification_accuracy = 0
                test_image_classification_accuracy = 0
            
            #if epoch % 40 == 0:
            print("__________________________")
            print("MEAN LOSS: ", mean_loss_new)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy_hard))
            print("IMAGE CLASSIFICATION: train accuracy : {}\ttest accuracy : {}".format(train_image_classification_accuracy,test_image_classification_accuracy))

            mean_loss = mean_loss_new
            sheduler.step(mean_loss)
        #write the accuracies of the last epoch

        self.classifier = best_classifier    


    def train_DFA(self, batch_size, num_of_epochs, decay=0.999, freezed=False):
        def get_lr(optim):
            for param_group in optim.param_groups:
                return param_group['lr']

        tot_size = len(self.train_traces)
        mean_loss = 1000000

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        dev_file = open(self.log_dir+self.ltl_formula_string+"_dev_acc_NS_exp"+str(self.exp_num), 'w')

        train_file_dfa = open(self.log_dir+self.ltl_formula_string+"_train_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        dev_file_dfa = open(self.log_dir+self.ltl_formula_string+"_dev_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        test_file_dfa = open(self.log_dir+self.ltl_formula_string+"_test_acc_dfa_NS_exp"+str(self.exp_num), 'w')
        loss_file = open(self.log_dir+self.ltl_formula_string+"_loss_dfa_NS_exp"+str(self.exp_num), 'w')

        cross_entr = torch.nn.CrossEntropyLoss()
        print("_____________training the DFA_____________")
        print("training on {} sequences using {} automaton states".format(tot_size, self.numb_of_states))

        params = [self.deepAutoma.trans_prob] + [self.deepAutoma.rew_matrix]
        optimizer = torch.optim.Adam(params, lr=0.01)
        sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-04)


        min_temp = 0.00001
        self.temperature =1.0

        if freezed:
            self.temperature = min_temp

        start_time = time.time()
        epoch= -1
        while True:
            epoch+=1
            print("epoch: ", epoch)
            losses = []
            for i in range(len(self.train_traces)):

                batch_trace_dataset = self.train_traces[i].to(device)
                batch_acceptance = self.train_acceptance_tr[i].to(device)
                optimizer.zero_grad()

                predictions= self.deepAutoma(batch_trace_dataset, self.temperature)

                loss = cross_entr(predictions, batch_acceptance)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=self.temperature)
            mean_loss_new = mean(losses)
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy : {}\tloss : {}".format(train_accuracy, test_accuracy, mean_loss_new))

            train_file.write("{}\n".format(train_accuracy))
            dev_file.write("{}\n".format(test_accuracy))
            train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='logic_circuit', temp=min_temp)
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy : {}".format(train_accuracy, test_accuracy))

            train_file_dfa.write("{}\n".format(train_accuracy))
            dev_file_dfa.write("{}\n".format(test_accuracy))
            loss_file.write("{}\n".format(mean(losses)))
            if freezed:
                self.temperature = min_temp
            else:
                self.temperature = max(self.temperature*decay, min_temp)
            print("temp: ", self.temperature)

            sheduler.step(mean_loss_new)
            print("lr: ", get_lr(optimizer))
            if mean_loss_new < 0.318 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            if epoch > 200 and abs(mean_loss_new - mean_loss) < 0.0001:
                break
            mean_loss = mean_loss_new


        ######################## net2dfa
        #save the minimized dfa
        self.dfa = self.deepAutoma.net2dfa( min_temp)
        ex_time =  time.time() - start_time

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+".ex_time", "w") as f:
            f.write("{}\n".format(ex_time))

        #print it
        try:
            self.dfa.to_graphviz().render("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_minimized.dot")
        except:
            print("Not able to render automa")
        with open("DFA_predicted_nesy/"+self.ltl_formula_string, 'wb') as outp:
            pickle.dump(self.dfa, outp, pickle.HIGHEST_PROTOCOL)

        with open("DFA_predicted_nesy/"+self.ltl_formula_string+"_exp"+str(self.exp_num)+"_min_num_states", "w") as f:
            f.write(str(len(self.dfa._states)))

        #ULTIMO TEST usando il DFA sul TEST set
        train_accuracy, test_accuracy = self.eval_learnt_DFA(automa_implementation='dfa', temp=min_temp, mode="test")
        print("FINAL SEQUENCE CLASSIFICATION ON TEST SET: {}".format(test_accuracy))

        test_file_dfa.write("{}\n".format(test_accuracy))

    def eval_all(self, automa_implementation, temperature, discretize_labels=False):
        train_accuracy = eval_acceptance(self.classifier, self.deepAutoma, self.alphabet, (self.train_img_seq, self.train_acceptance_img), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        test_accuracy_hard= eval_acceptance( self.classifier, self.deepAutoma, self.alphabet,(self.test_img_seq_hard, self.test_acceptance_img_hard), automa_implementation, temperature, discretize_labels=discretize_labels, mutually_exc_sym=True)

        return train_accuracy, 0,0, test_accuracy_hard

    def eval_image_classification(self):
        train_acc = eval_image_classification_from_traces(self.custom_trace, self.symbolic_grid, self.classifier, True)
        test_acc = train_acc
        return train_acc, test_acc