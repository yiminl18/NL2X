import random
from typing import Any, Dict, List, Optional
import json

from model.azuregpt4o import gpt_4o_azure
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig, ContentDataType
from . import register_benchmark
import os
import base64
import pandas as pd
import tiktoken

CUAD_PATH = "./benchmarks/CUAD/"
CUAD_DATA_PATH = CUAD_PATH + "CUAD_v1.json"

class CUADDataLoader:
    def __init__(self, data_path: str): 
        # Read CUAD_v1.json
        with open(data_path, 'r') as f:
            data = json.load(f)
        data = data['data']

        documents = []
        answers = []

        # questions = ''
        for item in data:
            real_doc = item['paragraphs'][0]
            documents.append(real_doc['context'])

            qas = real_doc['qas']
            local_answers = []
            for i, qa in enumerate(qas):
                # id = qa['id'].split('__')[-1]
                # questions += str(i + 1) + '. ' + id + ': ' + qa['question'].split('Details: ')[-1] + '\n'
                local_answer = []
                for answer in qa['answers']:
                    if answer['text']:
                        local_answer.append(answer['text'])
                local_answers.append(local_answer)

            answers.append(local_answers)
            # print('*' * 50)
            # print(questions)

        self.documents = documents
        self.answers = answers
        # print(self.answers[2])
        self.question = """For each document, extract the text spans (if they exist) for each of the following categories:
1. Document Name: The name of the contract
2. Parties: The two or more parties who signed the contract
3. Agreement Date: The date of the contract
4. Effective Date: The date when the contract is effective 
5. Expiration Date: On what date will the contract's initial term expire?
6. Renewal Term: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.
7. Notice Period To Terminate Renewal: What is the notice period required to terminate renewal?
8. Governing Law: Which state/country's law governs the interpretation of the contract?
9. Most Favored Nation: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?
10. Non-Compete: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector? 
11. Exclusivity: Is there an exclusive dealing  commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on  collaborating or working with other parties), whether during the contract or  after the contract ends (or both).
12. No-Solicit Of Customers: Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?
13. Competitive Restriction Exception: This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.
14. No-Solicit Of Employees: Is there a restriction on a party’s soliciting or hiring employees and/or contractors from the  counterparty, whether during the contract or after the contract ends (or both)?
15. Non-Disparagement: Is there a requirement on a party not to disparage the counterparty?
16. Termination For Convenience: Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?
17. Rofr/Rofo/Rofn: Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?
18. Change Of Control: Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?
19. Anti-Assignment: Is consent or notice required of a party if the contract is assigned to a third party?
20. Revenue/Profit Sharing: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?
21. Price Restrictions: Is there a restriction on the  ability of a party to raise or reduce prices of technology, goods, or  services provided?
22. Minimum Commitment: Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?
23. Volume Restriction: Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?
24. Ip Ownership Assignment: Does intellectual property created  by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?
25. Joint Ip Ownership: Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?
26. License Grant: Does the contract contain a license granted by one party to its counterparty?
27. Non-Transferable License: Does the contract limit the ability of a party to transfer the license being granted to a third party?
28. Affiliate License-Licensor: Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor? 
29. Affiliate License-Licensee: Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?
30. Unlimited/All-You-Can-Eat-License: Is there a clause granting one party an “enterprise,” “all you can eat” or unlimited usage license?
31. Irrevocable Or Perpetual License: Does the contract contain a  license grant that is irrevocable or perpetual?
32. Source Code Escrow: Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,  insolvency, etc.)?
33. Post-Termination Services: Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?
34. Audit Rights: Does a party have the right to  audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?
35. Uncapped Liability: Is a party’s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.
36. Cap On Liability: Does the contract include a cap on liability upon the breach of a party’s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.
37. Liquidated Damages: Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?
38. Warranty Duration: What is the duration of any  warranty against defects or errors in technology, products, or services  provided under the contract?
39. Insurance: Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?
40. Covenant Not To Sue: Is a party restricted from contesting the validity of the counterparty’s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?
41. Third Party Beneficiary: Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?

Texts must be extracted from documents, and each catogorization is wrapped by list. Here is a answer example:
[['SUPPLY CONTRACT'], ['The seller:', 'The buyer/End-User: Shenzhen LOHAS Supply Chain Management Co., Ltd.'], [], [], ['The Contract is valid for 5 years, beginning from and ended on .'], [], [], ["It will be governed by the law of the People's Republic of China ,otherwise it is governed by United Nations Convention on Contract for the International Sale of Goods."], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['Within 7 days after the arrival of the goods at destination, should the quality, specification, or quantity be found not in conformity with the stipulations of the Contract except those claims for which the insurance company or the owners of the vessel are liable, the Buyers, on the strength of the Inspection Certificate issued by the China Commodity Inspection Bureau, have the right to claim for replacement with new goods, or for compensation, and all the expenses (such as inspection charges, freight for returning the goods and for sending the replacement, insurance premium, storage and loading and unloading charges etc.) shall be borne by the Sellers.'], ['To be covered by the Seller for 110% invoice value against All Risks and War Risk.'], [], []]
"""

@register_benchmark("cuad")
class CUADBenchmark(BenchmarkInterface):
    def _setup(self):
        self.data_formats = ["txt"]
        if self.config.verbose:
            self.logger.info(f"Initialized CUAD benchmark with config: {self.config}")

    def _load_data(self) -> List[BenchmarkSample]:
        samples = []
        data_loader = CUADDataLoader(CUAD_DATA_PATH)
        num_samples = min(self.config.max_samples, len(data_loader.documents))
        # num_samples = min(self.config.max_samples, len(data_loader.documents))
        docs = []
        answers = []
        for idx in range(num_samples):
            doc = data_loader.documents[idx]
            answer = data_loader.answers[idx]
            docs.append({'src': doc})
            answers.append(answer)
            
                    
        context = {"CUAD_v1.json": ContentDataType("json", content = docs)}
        samples.append(BenchmarkSample(
                id='0',
                query=data_loader.question,
                context=context,
                ground_truth=answers
            ))
        self._samples = samples
        return self._samples

    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        # Check if prediction['answer'] is a list at the top level
        if not isinstance(prediction['answer'], list):
            return EvaluationResult(
                sample_id=sample.id,
                prediction=prediction,
                ground_truth=sample.ground_truth,
                metrics={"f1": 0.0, "recall": 0.0, "precision": 0.0}
            )

        # Clean prediction['answer']: ensure inner elements are lists
        cleaned_answer = []
        for item in prediction['answer']:
            if isinstance(item, list):
                # Filter out non-string elements from inner lists
                cleaned_inner = [elem for elem in item if isinstance(elem, str)]
                cleaned_answer.append(cleaned_inner)
            else:
                # Replace non-list items with empty lists
                cleaned_answer.append([])

        # Adjust length to match ground_truth
        gt_length = len(sample.ground_truth)
        pred_length = len(cleaned_answer)

        if pred_length > gt_length:
            # Truncate if prediction is longer
            cleaned_answer = cleaned_answer[:gt_length]
        elif pred_length < gt_length:
            # Pad with empty lists if prediction is shorter
            cleaned_answer.extend([[] for _ in range(gt_length - pred_length)])

        # Update prediction with cleaned answer
        prediction['answer'] = cleaned_answer

        # Compute the Jaccard similarity, correct if similarity > 0.15, compute precision and recall
        def jaccard_similarity(str1: str, str2: str) -> float:
            """Compute Jaccard similarity between two strings."""
            if not str1 and not str2:
                return 1.0
            if not str1 or not str2:
                return 0.0

            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))

            return intersection / union if union > 0 else 0.0

        total_correct = 0
        total_gt = 0
        total_pred = 0

        for gt_list, pred_list in zip(sample.ground_truth, prediction['answer']):
            if not isinstance(pred_list, list):
                pred_list = [pred_list] if pred_list else []
            if not isinstance(gt_list, list):
                gt_list = [gt_list] if gt_list else []

            total_gt += len(gt_list)
            total_pred += len(pred_list)

            if not gt_list or not pred_list:
                continue

            # Pre-compute all pairwise similarities with caching
            similarity_cache = {}
            for i, gt_item in enumerate(gt_list):
                for j, pred_item in enumerate(pred_list):
                    gt_str = str(gt_item)
                    pred_str = str(pred_item)
                    sim = jaccard_similarity(gt_str, pred_str)
                    similarity_cache[(i, j)] = sim

            # Track which items have been matched
            gt_matched = set()
            pred_matched = set()

            # Continue matching until no more matches possible
            while len(gt_matched) < len(gt_list) and len(pred_matched) < len(pred_list):
                # Find best available similarity from cache
                best_sim = 0.0
                best_pair = None

                for (i, j), sim in similarity_cache.items():
                    if i not in gt_matched and j not in pred_matched and sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

                # Check if best similarity is above threshold
                if best_pair and best_sim > 0.15:
                    gt_idx, pred_idx = best_pair
                    total_correct += 1
                    # Mark items as matched
                    gt_matched.add(gt_idx)
                    pred_matched.add(pred_idx)
                else:
                    # No more matches possible
                    break

        # Calculate metrics
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_gt if total_gt > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=sample.ground_truth,
            metrics={"f1": f1, "precision": precision, "recall": recall}
        )
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        aggregate = {
            "f1": sum(r.metrics["f1"] for r in results) / len(results),
            "precision": sum(r.metrics["precision"] for r in results) / len(results),
            "recall": sum(r.metrics["recall"] for r in results) / len(results),
        }
        
        return aggregate

if __name__ == "__main__":
    data_loader = CUADDataLoader('/Users/chiyuh/Workspace/NL2X/benchmarks/CUAD/CUAD_v1.json')