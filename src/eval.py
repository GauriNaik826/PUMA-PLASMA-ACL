from datasets import load_metric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score
import numpy as np
import pandas as pd
import argparse
import mlflow
from mlflow.tracking import MlflowClient

class EvaluationMetrics:
    def __init__(self, predictions, references):
        self.predictions = predictions
        self.references = references

    def compute_rouge_score(self):
        rouge = Rouge()
        rouge_l_f1, rouge_l_recall, rouge_l_precision = [], [], []
        rouge_1_f1, rouge_1_recall, rouge_1_precision = [], [], []
        rouge_2_f1, rouge_2_recall, rouge_2_precision = [], [], []
        for prediction, reference in zip(self.predictions, self.references):
            scores = rouge.get_scores(prediction, reference)[0]
            
            rouge_l_f1.append(scores["rouge-l"]["f"])
            rouge_l_recall.append(scores["rouge-l"]["r"])
            rouge_l_precision.append(scores["rouge-l"]["p"])
            
            rouge_1_f1.append(scores["rouge-1"]["f"])
            rouge_1_recall.append(scores["rouge-1"]["r"])
            rouge_1_precision.append(scores["rouge-1"]["p"])
            
            rouge_2_f1.append(scores["rouge-2"]["f"])
            rouge_2_recall.append(scores["rouge-2"]["r"])
            rouge_2_precision.append(scores["rouge-2"]["p"])

        results = {
            "rouge_l": {
                "f1": np.mean(rouge_l_f1) * 100 ,
                "recall": np.mean(rouge_l_recall) * 100,
                "precision": np.mean(rouge_l_precision) * 100
            },
            "rouge_1": {
                "f1": np.mean(rouge_1_f1) * 100,
                "recall": np.mean(rouge_1_recall) * 100,
                "precision": np.mean(rouge_1_precision) * 100
            },
            "rouge_2": {
                "f1": np.mean(rouge_2_f1) * 100,
                "recall": np.mean(rouge_2_recall) * 100,
                "precision": np.mean(rouge_2_precision) * 100
            }
        }
        
        return results

    def compute_meteor_score(self):
        
        meteor = load_metric('meteor')
        scores = []
        for prediction, reference in zip(self.predictions, self.references):
            score = meteor.compute(predictions=[prediction], references=[reference])
            #scores.append(score)
            scores.append(score["meteor"])

        average_meteor_score = np.mean(scores)
        
        return {"meteor": average_meteor_score}
    

    def compute_bleu_scores(self):
        bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []
        smoothie = SmoothingFunction().method4
        for prediction, reference in zip(self.predictions, self.references):
            reference = [reference.split()]  # BLEU score API requires tokenized sentences
            prediction = prediction.split()
            
            bleu_1.append(sentence_bleu(reference, prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie))
            bleu_2.append(sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
            bleu_3.append(sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
            bleu_4.append(sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

        results = {
            "bleu_1": np.mean(bleu_1),
            "bleu_2": np.mean(bleu_2),
            "bleu_3": np.mean(bleu_3),
            "bleu_4": np.mean(bleu_4)
        }
        
        return results
    
    def compute_bertscore(self, predictions, references, lang="en"):
       
        P, R, F1 = bert_score(predictions, references, lang=lang)
        return F1.mean().item()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, default="./generated/generated_result.csv")
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    parser.add_argument("--mlflow_experiment", type=str, default="puma-plasma-evaluation")
    parser.add_argument("--run_name", type=str, default="eval")
    parser.add_argument("--rouge_l_threshold", type=float, default=0.30,
                        help="Minimum ROUGE-L F1 required to promote model to Production")
    parser.add_argument("--model_name", type=str, default="puma-plasma-flant5")
    parser.add_argument("--model_version", type=str, default=None,
                        help="MLflow model version in Staging to evaluate (promotes to Production if gate passes)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    df = pd.read_csv(args.results_csv)
    predictions = df['PREDICTED'].tolist()
    references = df['ACTUAL OUTPUT'].tolist()
    eval_metrics = EvaluationMetrics(predictions, references)

    rouge_scores  = eval_metrics.compute_rouge_score()
    meteor_scores = eval_metrics.compute_meteor_score()
    bleu_scores   = eval_metrics.compute_bleu_scores()
    bert_f1       = eval_metrics.compute_bertscore(predictions, references)

    print("ROUGE scores:", rouge_scores)
    print("METEOR score:", meteor_scores)
    print("BLEU scores:",  bleu_scores)
    print("BERTScore F1:", bert_f1)

    # ── Log all metrics to MLflow ─────────────────────────────────────────
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_metrics({
            "rouge_l_f1":        rouge_scores["rouge_l"]["f1"],
            "rouge_l_recall":    rouge_scores["rouge_l"]["recall"],
            "rouge_l_precision": rouge_scores["rouge_l"]["precision"],
            "rouge_1_f1":        rouge_scores["rouge_1"]["f1"],
            "rouge_2_f1":        rouge_scores["rouge_2"]["f1"],
            "meteor":            meteor_scores["meteor"],
            "bleu_1":            bleu_scores["bleu_1"],
            "bleu_2":            bleu_scores["bleu_2"],
            "bleu_3":            bleu_scores["bleu_3"],
            "bleu_4":            bleu_scores["bleu_4"],
            "bertscore_f1":      bert_f1,
        })
        mlflow.log_param("rouge_l_threshold", args.rouge_l_threshold)

    # ── Quality gate: promote Staging → Production if ROUGE-L ≥ threshold ─
    rouge_l = rouge_scores["rouge_l"]["f1"] / 100  # convert back to 0-1 scale
    if args.model_version and rouge_l >= args.rouge_l_threshold:
        client = MlflowClient(tracking_uri=args.mlflow_uri)
        client.transition_model_version_stage(
            name=args.model_name,
            version=args.model_version,
            stage="Production",
            archive_existing_versions=True,  # demotes old Production to Archived
        )
        print(f"Quality gate PASSED (ROUGE-L={rouge_l:.3f} >= {args.rouge_l_threshold})")
        print(f"Model {args.model_name} v{args.model_version} promoted to Production")
    elif args.model_version:
        print(f"Quality gate FAILED (ROUGE-L={rouge_l:.3f} < {args.rouge_l_threshold})")
        print("Model stays in Staging. No Production promotion.")
