from dictionary import SOS
import rnn
from helpers import batchForTrain

def maskNLLLoss(inp,target,mask,device):
    total = mask.sum
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, total.item()

def train(inputVariable,lenghts,targetVariable,mask,maxTargetLen,encoder,decoder,embedding,encoderOptimizer,decoderOptimizer,batch,clip,device,maxLen = 10):
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    inputVariable = inputVariable.to(device)
    targetVariable = targetVariable.to(device)
    mask = mask.to(device)

    lenghts = lenghts.to("cpu")

    loss = 0
    print_losses = []
    n_totals = 0

    encoderOutput,encoderHidden = encoder(inputVariable,lenghts)

    decoderInput = torch.LongTensor([[SOS for _ in range(batch)]])
    decoderInput = decoderInput.to(device)

    decoderHidden = encoderHidden[:decoder.nLayers]
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for t in range(maxTargetLen):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutput)
            decoderInput = targetVariable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(maxTargetLen):
            decoderOutput, decoder_hidden = decoder(decoderInput, decoderHidden, encoder_outputs)
            _, topi = decoderOutput.topk(1)
            decoderInput = torch.LongTensor([[topi[i][0] for i in range(batch)]])
            decoderInput = decoderInput.to(device)
            mask_loss, nTotal = maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoderOptimizer.step()
    decoderOptimizer.step()
        
    return sum(print_losses) / n_totals

def trainIters(modelName, dic, pairs, encoder, decoder, encoderOptimizer, decoderOptimizer, embedding, encoderNLayers, decoderNLayers, save_dir, n_iteration, batch, printEvery, saveEvery, clip, dataSubFolder, loadFilename):

    training_batches = [batchForTrain(dic, [random.choice(pairs) for _ in range(batch)])for _ in range(n_iteration)]

    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,decoder, embedding, encoderOptimizer, decoderOptimizer, batch, clip)
        print_loss += loss


        if iteration % printEvery == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % saveEvery == 0):
            directory = os.path.join(save_dir, modelName, dataSubFolder, '{}-{}_{}'.format(encoderNLayers, decoderNLayers, hiddenSize))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoderOptimizer.state_dict(),
                'de_opt': decoderOptimizer.state_dict(),
                'loss': loss,
                'dic_dict': dic.__dict__,
                'embedding': embedding.state_dict()}, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
