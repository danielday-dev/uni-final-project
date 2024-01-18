    #pytorch imports
import torch
import torch.nn as nn
import torchaudio
import torch.functional as f

#other file imports
import settings as s
import audio_dataset
import random
import log_spectral_distance as lsd
import time

class Seq2seq(nn.Module):
    '''seq2seq model for audio super resolution'''
    def __init__(self, input_size, output_size, hidden_size, n_layers, enc_dropout, dec_dropout, device, teacher_force_ratio = 0.5):
        super().__init__()
        
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.teacher_force_ratio = teacher_force_ratio        
        
        self.enc = nn.LSTM(input_size, hidden_size, n_layers, 
                           dropout = enc_dropout, batch_first = True)
        self.dec = nn.LSTM(output_size, hidden_size, n_layers, 
                           dropout = dec_dropout, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        
        
    def forward(self, inputs, targets):
        target_length = targets.shape[1]

        # Initialize outputs tensor with the correct shape
        outputs = torch.zeros(target_length, s.BATCH_SIZE, self.output_size).to(inputs.device)


        enc_out, (h, c) = self.enc(inputs)

        dec_in = targets[:, 0:1, :]

        for t in range(1, targets.shape[1]):
            # print(dec_in.size())
            out, (h, c) = self.dec(dec_in, (h, c))
            pred = self.fc(out)
            # print(pred.size())
            # print(outputs.size())
            outputs[t] = torch.transpose(pred,0,1)


            teacher_force = (random.random() >= self.teacher_force_ratio) and self.training
            
            # top1 = pred.softmax(1)
            # print(top1)
            # dec_in = targets[t] if teacher_force else top1
            dec_in = targets[:, t:t+1, :] if teacher_force else pred
            # print("new dec_in: ", dec_in)
            # print(targets[t], pred)
            # if t % 1000 == 0:
            #     print(t)
        outputs = torch.transpose(outputs, 0, 1)
        return outputs
    
    
def main():
    #set seed for reproduction
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    #setup device as graphics card if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print("Program started using: ", device)

    #setup dataloaders
    train_dataloader, val_dataloader, test_dataloader = audio_dataset.load_audio_dataset()
    print("Dataloaders created successfully")
    
    model = Seq2seq(s.INPUT_SIZE, s.OUTPUT_SIZE, s.HIDDEN_SIZE, s.N_LAYERS, s.ENC_DROPOUT, s.DEC_DROPOUT, device).to(device)
    print("seq2seq model loaded")
    
    #use cross entropy loss as criterion
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    #use adam as optimizer
    # print([i for i in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr = s.LEARNING_RATE)

    best_val = float('inf')
    print("load model (input 1) or create new model (input 2)")
    mode = input("--> ")

    if mode == "2":
        print("Starting Training")
        
        for epoch in range(s.EPOCHS): #TODO CHANGE TO ENUMERATE?
            start_time = time.time()
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            print("validating...")
            val_loss = validate(model, val_dataloader, criterion, device)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'Epoch {epoch+1}/{s.EPOCHS}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, - Time: {epoch_time:.2f} seconds')

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), 'model_seq2seq_after_epoch' + str(epoch) +  '.pt')


        torch.save(model.state_dict(), 'model_seq2seq.pt')
    elif mode == "1":
        model.load_state_dict(torch.load("model_seq2seq.pt"))
        print("model loaded successfully")
    final_test(model, test_dataloader, device)
    
    
def train(model, loader, crit, opt, device):
    '''Train one epoch, returns training loss'''   
    
    #tell the model it is being trained 
    model.train()

    epoch_loss = 0
    for i, inputs in enumerate(loader): #pad the batch? #also pack the batch in the dataloader? pack somewhere? flatten something?
        
        #send low and high quality to device
        # print(inputs)
        start_time = time.time()
        #seperate the target and the inputs
        low_quality = inputs[0].to(device)
        high_quality = inputs[1].to(device)
        
        # print("low_quality from dataset", low_quality.size())
        
        # low_quality = torch.transpose(low_quality, 2, 1)
        # high_quality = torch.transpose(high_quality, 2, 1)

        # print("low quality after transposing", low_quality.size())

        #reset optimizer for batch
        opt.zero_grad()

        #forward pass on model
        out = model(low_quality, high_quality)
        
        #compare outputs with desired high quality output to get loss
        # print("calulating loss")
        loss = crit(out, high_quality)
        # print(loss)
        #adjust weights based on loss function
        
        # loss.requires_grad = True
        # print("performing backprop")
        loss.backward()
        
        # print("stepping opt")
        opt.step()

        #increment the loss
        epoch_loss += loss.item()
        end_time = time.time()
        batch_time = end_time - start_time
        print("finished batch number", i+1, ", it took this many seconds:", batch_time)
        #update loss every batch
        # print(epoch_loss)
        # print(loss.item())
    return epoch_loss / len(loader)

def validate(model, loader, crit, device):
    '''Validate one epoch, returns validation loss'''
    
    #tell model it is being evaluated
    model.eval()
    total_loss = 0

    #do not apply gradient
    with torch.no_grad():
        for inputs in loader:
            # send low and high quality to device
            low_quality = inputs[0].to(device)
            high_quality = inputs[1].to(device)

            #transpose inputs
            # low_quality = torch.transpose(low_quality, 2, 1)
            # high_quality = torch.transpose(high_quality, 2, 1)
            # forward pass on model
            out = model(low_quality, high_quality)

            # compare expected high quality and model's values
            loss = crit(out, high_quality)

            total_loss += loss.item()

    avg_loss = total_loss/ s.BATCH_SIZE
    return avg_loss

def final_test(model, test_dataloader, device):
    '''performs final tests on model'''
    print("beginning final tests")
    with torch.no_grad():
        i = 1
        for inputs in test_dataloader:
            low_quality = inputs[0].to(device)
            high_quality = inputs[1].to(device)
            
            
            # low_quality = torch.transpose(low_quality, 2, 1)
            # high_quality = torch.transpose(high_quality, 2, 1)
            
            model.eval()
            waveform = model(low_quality, high_quality)

            waveform = waveform.cpu()

            for batched_input_idx in range(0, s.BATCH_SIZE):
                wave = torch.transpose(waveform[batched_input_idx], 0 ,1)
                # print(wave.size())
                torchaudio.save(
                    "./final_out/audio_test_"+str(i)+".wav", 
                    wave, s.HQ_SAMPLE_RATE,
                    encoding="PCM_S")
                          
                wave = wave.tolist()
                
                hq = high_quality[batched_input_idx]
                hq = torch.transpose(hq, 0, 1)
                hq = hq.cpu().tolist()
                
                print("LSD for wave:", i, "=", lsd.log_spectral_distance(wave, hq, sr=s.HQ_SAMPLE_RATE))
                i = i + 1  


if __name__ == "__main__":
    main()