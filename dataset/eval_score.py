import json
import os, sys
import glob
import subprocess

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install("evaluate")

import evaluate 
bleu_score = evaluate.load("bleu")
chrf_score = evaluate.load("chrf")
comet_score = evaluate.load("comet")

def load_labels(filename):
	with open(filename, encoding="utf-8") as f:
		labels = f.readlines()
	labels = [x.strip() for x in labels]
	return labels

def split_sequence(sequence, chunk_size):
    chunks=[]
    for i in range(0, len(sequence), chunk_size):
        chunks.append(sequence[i: i + chunk_size])
    return chunks
		
def print_usage():
    print (
    '''Usage:
    python3 NADI2024-ST3-Scorer.py NADI2024_subtask3_dev_GOLD.txt pred_file.txt UBC_subtask3_dev_1.txt
        ''')


if __name__ == '__main__':
    
    truth_file = sys.argv[1]
    prediction_file = sys.argv[2]
    source_file = sys.argv[3]

    truth = load_labels(truth_file)
    truth = [t.strip() for t in truth]

    prediction = load_labels(prediction_file)
    prediction = [p.strip() for p in prediction]

    source = load_labels(source_file)
    source = [s.strip() for s in source]
    
    if (len(truth) != len(prediction)):
        print ("both files must have same number of instances")
        exit()

    chunk_size=100 
    truth_chunks= split_sequence(truth, chunk_size)

    truth_Egyptain=truth_chunks[0]
    truth_Emirati=truth_chunks[1]
    truth_Jordanian=truth_chunks[2]
    truth_Palestinian=truth_chunks[3]

    prediction_chunks= split_sequence(prediction, chunk_size)

    prediction_Egyptain=prediction_chunks[0]
    prediction_Emirati=prediction_chunks[1]
    prediction_Jordanian=prediction_chunks[2]
    prediction_Palestinian=prediction_chunks[3]

    source_chunks= split_sequence(prediction, chunk_size)

    source_Egyptain=source_chunks[0]
    source_Emirati=source_chunks[1]
    source_Jordanian=source_chunks[2]
    source_Palestinian=source_chunks[3]

    ### get scores
    bleu_Egyptain = bleu_score.compute(predictions=prediction_Egyptain, references=truth_Egyptain)['bleu']*100
    chrf_Egyptain = chrf_score.compute(predictions=prediction_Egyptain, references=truth_Egyptain)['score']
    comet_Egyptain = comet_score.compute(predictions=prediction_Egyptain, references=truth_Egyptain, sources=source_Egyptain)['mean_score']*100
    
    bleu_Emirati = bleu_score.compute(predictions=prediction_Emirati, references=truth_Emirati)['bleu']*100
    chrf_Emirati = chrf_score.compute(predictions=prediction_Emirati, references=truth_Emirati)['score']
    comet_Emirati = comet_score.compute(predictions=prediction_Emirati, references=truth_Emirati, sources=source_Emirati)['mean_score']*100

    bleu_Jordanian = bleu_score.compute(predictions=prediction_Jordanian, references=truth_Jordanian)['bleu']*100
    chrf_Jordanian = chrf_score.compute(predictions=prediction_Jordanian, references=truth_Jordanian)['score']
    comet_Jordanian = comet_score.compute(predictions=prediction_Jordanian, references=truth_Jordanian, sources=source_Jordanian)['mean_score']*100
    
    bleu_Palestinian = bleu_score.compute(predictions=prediction_Palestinian, references=truth_Palestinian)['bleu']*100
    chrf_Palestinian = chrf_score.compute(predictions=prediction_Palestinian, references=truth_Palestinian)['score']
    comet_Palestinian = comet_score.compute(predictions=prediction_Palestinian, references=truth_Palestinian, sources=source_Palestinian)['mean_score']*100
    
    bleu_all = bleu_score.compute(predictions=prediction, references=truth)['bleu']*100
    chrf_all = chrf_score.compute(predictions=prediction, references=truth)['score']
    comet_all = comet_score.compute(predictions=prediction, references=truth, sources=source)['mean_score']*100


    #write to a text file
    print('Scores:')
    scores = {
              'Overall': {
                'bleu': bleu_all,
                'chrf': chrf_all,
                'comet': comet_all
              },
              'Egyptain': {
                'bleu': bleu_Egyptain,
                'chrf': chrf_Egyptain,
                'comet': comet_Egyptain
              },
              'Emirati': {
                'bleu': bleu_Emirati,
                'chrf': chrf_Emirati,
                'comet': comet_Emirati
              },
              'Jordanian': {
                'bleu': bleu_Jordanian,
                'chrf': chrf_Jordanian,
                'comet': comet_Jordanian
              },
              'Palestinian': {
                'bleu': bleu_Palestinian,
                'chrf': chrf_Palestinian,
                'comet': comet_Palestinian
              },
             }
    print(scores)

    # with open(prediction_file.split("/")[-1].split(".")[0]+"_result.txt", 'w') as score_file:
    #     score_file.write("Overall: %0.12f\n" % scores["Overall"])
    #     score_file.write("Egyptain: %0.12f\n" % scores["Egyptain"])
    #     score_file.write("Emirati: %0.12f\n" % scores["Emirati"])
    #     score_file.write("Jordanian: %0.12f\n" % scores["Jordanian"])
    #     score_file.write("Palestinian: %0.12f\n" % scores["Palestinian"])

    
   
