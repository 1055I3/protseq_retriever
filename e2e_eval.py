import os
import json
import time
import asyncio
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

# Import the interface to call the pipeline
from pipeline_interface import run_pipeline_interface

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TestCase:
    id: int
    name: str
    prompt: str
    expected_type: str
    expected_accessions: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    is_negative: bool = False

# =============================================================================
# HELPER FUNCTIONS (METRICS & EVALUATION)
# =============================================================================

def calculate_rr(matches: List[Dict[str, Any]], expected_accessions: List[str]) -> float:
    """Calculates Reciprocal Rank (RR)."""
    if not expected_accessions:
        return 0.0
    for i, match in enumerate(matches):
        if match.get("primaryAccession") in expected_accessions:
            return 1.0 / (i + 1)
    return 0.0

def calculate_recall_at_k(matches: List[Dict[str, Any]], expected_accessions: List[str], k: int) -> float:
    """Calculates Recall@K (Binary: 1 if any expected found in top K, else 0)."""
    if not expected_accessions:
        return 0.0
    top_k_accs = [m.get("primaryAccession") for m in matches[:k]]
    for acc in expected_accessions:
        if acc in top_k_accs:
            return 1.0
    return 0.0

def calculate_ndcg(matches: List[Dict[str, Any]], expected_accessions: List[str], k: int) -> float:
    """Calculates simplified nDCG@K (Binary relevance: 1 for match, 0 otherwise)."""
    if not expected_accessions:
        return 0.0
    
    dcg = 0.0
    for i, match in enumerate(matches[:k]):
        rel = 1.0 if match.get("primaryAccession") in expected_accessions else 0.0
        dcg += rel / math.log2(i + 2)
    
    # IDCG for binary relevance: 1 if at least one relevant item exists, else 0
    # Simplified: since we usually expect 1 primary hit, IDCG is 1/log2(2) = 1.0
    idcg = 1.0 
    return dcg / idcg

def evaluate_constraints(matches: List[Dict[str, Any]], constraints: Dict[str, Any]) -> float:
    """
    Evaluates structured biological constraints against retrieval results.
    Returns a score from 0.0 to 1.0 representing the fraction of satisfied constraints.
    """
    if not constraints or not matches:
        return 1.0
    
    total_constraints = len(constraints)
    satisfied_count = 0
    
    # We evaluate against the Top 1 match for constraint strictness
    top_match = matches[0]
    
    # Flatten match for keyword fallback
    record_text = json.dumps(top_match).lower()
    
    for c_type, c_values in constraints.items():
        is_satisfied = False
        
        if c_type == "include_taxa":
            lineage = top_match.get("organism", {}).get("lineage", [])
            org_name = top_match.get("organism", {}).get("scientificName", "").lower()
            if any(v.lower() in org_name or any(v.lower() in l.lower() for l in lineage) for v in c_values):
                is_satisfied = True
        
        elif c_type == "exclude_taxa":
            lineage = top_match.get("organism", {}).get("lineage", [])
            org_name = top_match.get("organism", {}).get("scientificName", "").lower()
            if not any(v.lower() in org_name or any(v.lower() in l.lower() for l in lineage) for v in c_values):
                is_satisfied = True
                
        elif c_type == "subcellular_location":
            locations = []
            for comment in top_match.get("comments", []):
                if comment.get("commentType") == "SUBCELLULAR_LOCATION":
                    for loc in comment.get("subcellularLocations", []):
                        val = loc.get("location", {}).get("value", "").lower()
                        if val: locations.append(val)
            if any(v.lower() in " ".join(locations) for v in c_values):
                is_satisfied = True
                
        elif c_type == "ec_numbers":
            ec_nums = []
            for xref in top_match.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "EC":
                    ec_nums.append(xref.get("id", ""))
            if any(v in ec_nums for v in c_values):
                is_satisfied = True
                
        elif c_type == "interaction_partners":
            # Heuristic check in interaction comments
            partners = []
            for comment in top_match.get("comments", []):
                if comment.get("commentType") == "INTERACTION":
                    # Note: structure varies, fallback to text scan
                    pass
            if any(v.lower() in record_text for v in c_values):
                is_satisfied = True
                
        elif c_type == "functional_terms":
            if any(v.lower() in record_text for v in c_values):
                is_satisfied = True
                
        if is_satisfied:
            satisfied_count += 1
            
    return satisfied_count / total_constraints

# =============================================================================
# EVALUATOR ENGINE
# =============================================================================

class BioSeqEvaluator:
    def __init__(self):
        self.test_cases = [
            # --- Legacy / Core ---
            TestCase(1, "Direct Protein Identification", 
                     "Identify this sequence: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN", 
                     "PROTEIN", ["P01308"]),
            
            TestCase(2, "Taxonomic Exclusion", 
                     "Find sequences similar to human insulin (MALWMRLL...) but I am interested in any species except Human.", 
                     "PROTEIN", ["P01315", "P01317"], {"exclude_taxa": ["Homo sapiens"]}),
            
            TestCase(3, "Functional context (Glucose)", 
                     "I have this protein [MALWMRLL...]. Is it involved in glucose metabolism?", 
                     "PROTEIN", ["P01308"], {"functional_terms": ["glucose metabolism"]}),
            
            TestCase(4, "DNA Sequence Routing", 
                     "Consider this gene sequence: ATGACAAAGAGACTCCGGGTGGAAGATGACTTCAACCCCGTCTACCCCTATGGCTACGCGCGGAATCAAAATATTCCCTTCCTCACTCCCCCCTTTGTCTCCTCCAATGGATTTCAAAACTTCCCCCCTGGGGTCCTGTCACTTAAACTGGCTGACCCAATCACCATTAACAATCAAAATGTATCACTCAAGGTTGGAGGGGGGCTAACTTTGCAAGAAGAAACTGGAAAATTAACAGTTAATACTGAACCACCTTTGCATCTTACAAATAACAAATTAGGGATAGCTTTAGACGCTCCATTTGATGTTATAGACAATAAGCTTACACTATTAGCAGGCCATGGCTTGTCTATTATAACAAAAGAAACATCAACACTGCCTGGCTTGGTTAATACTCTTGTAGTATTAACTGGAAAGGGTATTGGAACAGATTTATCAAATAATGGTGGAAATATATGTGTTAGAGTTGGAGAAGGCGGCGGCTTATCATTTAATGACAATGGAGACTTGGTAGCATTTAATAAAAAAGAAGACAAACGCACCCTATGGACAACTCCAGACACATCTCCAAATTGCAGAATTGATCAGGATAAGGACTCTAAGCTAACTTTGGTCCTTACAAAGTGTGGAAGTCAAATATTAGCCAATGTGTCATTAATTGTTGTAGCTGGAAGGTACAAAATTATCAATAACAATACTAATCCAGCTCTTAAAGGATTTACCATTAAATTGTTGTTTGATAAAAATGGAGTCCTTATGGAATCTTCAAATCTTGGTAAATCATATTGGAACTTTCGAAATCAAAATTCAATTATGTCAACAGCTTATGAAAAAGCTATTGGTTTTATGCCTAATTTGGTAGCCTATCCAAAACCTACCACTGGCTCTAAAAAATATGCAAGAGATATAGTTTATGGAAACATCTACCTTGGCGGAAAGCCACATCAACCAGTAACCATTAAAACTACCTTTAACCAGGAAACTGGATGTGAATACTCTATTACATTTGATTTTAGTTGGGCCAAAACTTATGTAAATGTTGAATTTGAAACTACCTCTTTTACCTTTTCCTATATTGCCCAAGAATAA. What is it?", 
                     "DNA", ["P53110"]),

            # --- Diverse Protein Families ---
            TestCase(5, "GPCR Example (Rhodopsin)", 
                     "Find matches for this G protein-coupled receptor involved in visual phototransduction: MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTTTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFXWVMALACAAPPLVGWSRYIPEGMQCSCGIDYYTPHEETNNESFVIYMFVVHFIIPLIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWLPYAGVAFYIFTHQGSDFGPIFMTIPAFFAKTSAVYNPVIYIMMNKQFRNCMVTTLCCGKNPLGDDEASTTVSKTETSQVAPA",
                     "PROTEIN", ["P08100"], {"functional_terms": ["phototransduction", "GPCR"]}),

            TestCase(6, "Kinase Example (ABL1)", 
                     "Search for this tyrosine-protein kinase: MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVAXGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGRLAKLKPAPPPPPAAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR",
                     "PROTEIN", ["P00519"], {"ec_numbers": ["2.7.10.2"]}),

            TestCase(7, "Membrane Protein (UNC5C)", 
                     "I have a sequence for a netrin receptor involved in axon guidance. Is it UNC5C? Seq: MARAGSGAAGGRAGGAGRAAWPGLRALLGLLLPGVTAAAMNGVPTAEEVSPKPDLTVALNREVARSLSCTVTGHPKPVVSWQKDERPLDNGHYLVRNSHGLNILRIQNARPGDNGIYVCSASNPVGRQSTXTRLRVQEIDTPLPQEVEIKEVEEAYVPCVATHPQPQITWQKNGRPFADKGYYVTESNRLLVLELSNAKPDDMGLYVCSANNPIGEQSSTSRLRVQEVDSPDPKLSYKVVDEGRPVPCVAGHPVPDVTWQKNGVPFSDKGYLVLENSHGLRILELSRANPGDMGHYVCSANNPVGEQSSTSRLRVQEVDTPLPQEVEVKEVEEAAVPCVATHPQPQITWQKNGRPFADKGYYVTESNRLLVLELSNAKPDDMGLYVCSANNPIGEQSSTSRLRVQEVDSPDPKLSYKVVDEGRPVPCVAGHPVPDVTWQKNGVPFSDKGYLVLENSHGLRILELSRANPGDMGHYVCSANNPVGEQSSTSRLRVQEVDTPVPKVDVKEVEEAAVPCVATHPQPQITWQKNGRPFADKGYYVTESNRLLVLELSNAKPDDMGLYVCSANNPIGEQSSTSRLRVQEVDSPDPKLSYKVVDEGRPVPCVAGHPVPDVTWQKNGVPFSDKGYLVLENSHGLRILELSRANPGDMGHYVCSANNPVGEQSSTSRLRVQEVDTPVPKVDVKEVEEAAVPCVATHPQPQITWQKNGRPFADKGYYVTESNRLLVLELSNAKPDDMGLYVCSANNPIGEQSSTSRLRVQEVDSPDPKLSYKVVDEGRPVPCVAGHPVPDVTWQKNGVPFSDKGYLVLENSHGLRILELSRANPGDMGHYVCSANNPVGEQSSTSRLRVQEVD",
                     "PROTEIN", ["O95185"], {"subcellular_location": ["membrane"], "functional_terms": ["axon guidance"]}),

            TestCase(8, "Bacterial Enzyme (Beta-lactamase)", 
                     "Search for this bacterial enzyme sequence: MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAXWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
                     "PROTEIN", ["P62593"], {"include_taxa": ["Bacteria"], "ec_numbers": ["3.5.2.6"]}),

            TestCase(9, "Viral Protein (SARS-CoV-2 Spike S1)", 
                     "Identify this viral protein fragment from SARS-CoV-2: MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYXENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARS",
                     "PROTEIN", ["P0DTC2"], {"include_taxa": ["Viruses"]}),

            # --- Length Stratified / Edge Cases ---
            TestCase(10, "Short Peptide (Oxytocin)", 
                     "Sequence: CYIQNCPLG", 
                     "PROTEIN", ["P01178"]),
            
            TestCase(11, "Fragmented Sequence", 
                     "What is this partial sequence: MALT...GIVEQCCTSICSLYQLENYCN", 
                     "PROTEIN", ["P01308"], {"functional_terms": ["insulin"]}),

            TestCase(12, "Logical Firewall (Negative)", 
                     "How do I bake a chocolate cake?", 
                     "ERROR", [], {}, True),

            # --- New DNA Test Cases ---
            TestCase(13, "Viral DNA (Adenovirus Fiber)",
                     "Analyze this DNA sequence from a human adenovirus: ATGCAGCGCGCGGCGATGTATGAGGAAGGTCCTCCTCCCTCCTACGAGAGCGTGGTGAGCGCGGCGCCAGTGGCGGCGGCGCTGGGTTCCCCCTTCGATGCTCCCCTGGACCCGCCGTTTGTGCCTCCGCGGTACCTGCGGCCTACCGGGGGGAGAAACAGCATCCGTTACTCTGAGTTGGCACCCCTATTCGACACCACCCGTGTGTACCTTGTGGACAACAAGTCAACGGATGTGGCATCCCTGAACTACCAGAACGACCACAGCAACTTTCTAACCACGGTCATTCAAAACAATGACTACAGCCCGGGGGAGGCAAGCACACAGACCATCAATCTTGACGACCGTTCGCACTGGGGCGGCGACCTGAAAACCATCCTGCATACCAACATGCCAAATGTGAACGAGTTCATGTTTACCAATAAGTTTAAGGCGCGGGTGATGGTGTCGCGCTCGCTTACTAAGGACAAACAGGTGGAGCTGAAATATGAGTGGGTGGAGTTCACGCTGCCCGAGGGCAACTACTCCGAGACCATGACCATAGACCTTATGAACAACGCGATCGTGGAGCACTACTTGAAAGTGGGCAGGCAGAACGGGGTTCTGGAAAGCGACATCGGGGTAAAGTTTNACACCCGCAACTTCAGACTGGGGTTTGACCCAGTCACTGGTCTTGTCATGCCTGGGGTATATACAAACGAAGCCTTCCATCCAGACATCATTTTGCTGCCAGGATGCGGGGTGGACTTCACCCACAGCCGCCTGAGCAACTTGTTGGGCATCCGCAAGCGGCAACCCTTCCAGGAGGGCTTTAGGATCACCTACGATGACCTGGAGGGTGGTAACATTCCCGCACTGTTGGATGTGGACGCCTACCAGGCAAGCTTAAAAGATGACACCGAACAGGGCGGGGATGGCGCAGGCGGCGGCAACAACAGTGGCAGCGGCGCGGAAGAGAACTCCAACGCGGCAGCCGCGGCAATGCAGCCGGTGGAGGACATGAACGATCATGCCATTCGCGGCGACACCTTTGCCACACGGGCGGAGGAGAAGCGCGCTGAGGCCGAGGCAGCGGCAGAAGCTGCCGCCCCCGCTGCGCAACCCGAGGTCGAGAAGCCTCAGAAGAAACCGGTGATCAAACCCCTGACAGAGGACAGCAAGAAACGCAGTTACAACCTAATAAGCAATGACAGCACCTTCACCCAGTACCGCAGCTGGTACCTTGCATACAACTACGGCGACCCTCAGACCGGGATCCGCTCATGGACCCTCCTTTGCACTCCTGACGTAACCTGCGGCTCGGAGCAGGTCTACTGGTCGTTGCCAGACATGATGCAAGACCCCGTGACCTTCCGCTCCACGAGCCAGATCAGCAACTTTCCGGTGGTGGGCGCCGAGCTGTTGCCCGTGCACTCCAAGAGCTTCTACAACGACCAGGCCGTCTACTCCCAGCTCATCCGCCAGTTTACCTCTCTGACCCACGTGTTCAATCGCTTTCCCGAGAACCAGATTTTGGCGCGCCCGCCAGCCCCCACCATCACCACCGTCAGTGAAAACGTTCCTGCTCTCACAGATCACGGGACGCTACCGCTGCGCAACAGCATCGGAGGAGTCCAGCGAGTGACCATTACTGACGCCAGACGCCGCACCTGCCCCTACGTTTACAAGGCCCTGGGCATAGTCTCGCCGCGCGTCCTATCGAGCCGCACTTTTTGA Is it involved in cell attachment?",
                     "DNA", ["P03276"], {"include_taxa": ["Adenoviridae"], "functional_terms": ["attachment"]}),

            TestCase(14, "Bacterial Gene (recA)",
                     "Consider this bacterial DNA sequence: ATGGCTATCGACGAAAACAAACAGAAAGCGTTGGCGGCAGCACTGGNCCAGATTGAGAAACAATTTGGTAAAGGCTCCATCATGCGCCTGGGTGAAGACCGTTCCATGGATGTGGAAACCATCTCTACCGGTTCGCTTTCACTGGATATCGCGCTTGGGGCAGGTGGTCTGCCGATGGGCCGTATCGTCGAAATCTACGGACCGGAATCTTCCGGTAAAACCACGCTGACGCTGCAGGTGATCGCCGCAGCGCAGCGTGAAGGTAAAACCTGTGCGTTTATCGATGCTGAACACGCGCTGGACCCAATCTACGCACGTAAACTGGGCGTCGATATCGACAACCTGCTGTGCTCCCAGCCGGACACCGGCGAGCAGGCACTGGAAATCTGTGACGCCCTGGCGCGTTCTGGCGCAGTAGACGTTATCGTCGTTGACTCCGTGGCGGCACTGACGCCGAAAGCGGAAATCGAAGGCGAAATCGGCGACTCTCACATGGGCCTTGCGGCACGTATGATGAGCCAGGCGATGCGTAAGCTGGCGGGTAACCTGAAGCAGTCCAACACGCTGCTGATCTTCATCAACCAGATCCGTATGAAAATTGGTGTGATGTTCGGTAACCCGGAAACCACTACCGGTGGTAACGCGCTGAAATTCTACGCCTCTGTTCGTCTCGACATCCGTCGTATCGGCGCGGTGAAAGAGGGCGAAAACGTGGTGGGTAGCGAAACCCGCGTGAAAGTGGTGAAGAACAAAATCGCTGCGCCGTTTAAACAGGCTGAATTCCAGATCCTCTACGGCGAAGGTATCAACTTCTACGGCGAACTGGTTGACCTGGGCGTAAAAGAGAAGCTGATCGAGAAAGCAGGCGCGTGGTACTACTCCAAGGGTGAAAAAATCGGTCAGGGTAAAGCGAATGCGACTGCCTGGCTGAAAGATAACCCGGAAACCGCGAAAGAGATCGAGAAGAAAGTACGTGAGTTGCTGCTGAGCAACCCGAACTCAACGCCGGATTTCTCTGTAGATGATAGCGAAGNCGTAGCAGAAACTAACGAAGATTTTTAA Does it belong to RecA?",
                     "DNA", ["P0A7G6"], {"include_taxa": ["Bacteria"], "functional_terms": ["RecA", "DNA repair"]}),

            TestCase(15, "Mitochondrial DNA (cytochrome b)",
                     "What is this mitochondrial gene sequence: ATGACTCCACAAAGAAACACCAACCCCTTGAAGATGCTCATCAACCATTCTTTCATTGACCTCCCNCCTTCCAACATCTCTGCCTGACTCCCTACCCCATCCAACATCTCCGCATTCATGAAACTTCGGCTCACTCCTTGGCGCCTAGCATCATCATCATTCTACTGACTGGCCTAGCCCTCATCACCATCGCTAACCCCATGGCCATCATGTTCCTTCTACTCATCGGCGTGCTACTAACATGATCGTCACCCTCGTCTGTTTGCTAGGTGTGTGACTGCTCCTCCTAGCCATCGCTNTCATTACCGGCTTCATCCTAGCCCTAGCCCTACTACTAATCGCAATGGCCTTCCGCCCTATC?",
                     "DNA", ["P00156"], {"subcellular_location": ["Mitochondrion"], "functional_terms": ["cytochrome b"]})
        ]
        self.results = []
        self.metrics = {}

    def _evaluate_results(self, test: TestCase, output: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates comprehensive metrics for a single test case."""
        res = {
            "routing_ok": False,
            "ground_truth_found": False,
            "rr": 0.0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "constraint_score": 0.0,
            "failure_type": "UNKNOWN_FAILURE",
            "confidences": {},
            "status": "FAILED"
        }

        # 0. Capture Confidence if present
        res["confidences"] = {
            "routing": output.get("routing_confidence"),
            "retrieval": output.get("retrieval_confidence"),
            "rejection": output.get("rejection_confidence")
        }

        # 1. Routing Check
        if test.is_negative:
            res["routing_ok"] = "error" in output
            if not res["routing_ok"]:
                res["failure_type"] = "ROUTING_FAILURE"
        else:
            res["routing_ok"] = output.get("sequence_type") == test.expected_type
            if not res["routing_ok"]:
                res["failure_type"] = "ROUTING_FAILURE"

        # 2. Ranking Metrics
        raw_matches = output.get("results", [])
        # Ensure matches are sorted by the unified _search_score
        matches = sorted(raw_matches, key=lambda x: x.get("_search_score", 0), reverse=True)

        if not test.is_negative and matches:
            res["rr"] = calculate_rr(matches, test.expected_accessions)
            res["recall_at_1"] = calculate_recall_at_k(matches, test.expected_accessions, 1)
            res["recall_at_5"] = calculate_recall_at_k(matches, test.expected_accessions, 5)
            res["ndcg_at_5"] = calculate_ndcg(matches, test.expected_accessions, 5)
            res["ground_truth_found"] = res["recall_at_5"] > 0
            
            if not res["ground_truth_found"] and test.expected_accessions:
                res["failure_type"] = "RETRIEVAL_FAILURE"

        # 3. Structured Constraint Evaluation
        if not test.is_negative and matches:
            res["constraint_score"] = evaluate_constraints(matches, test.constraints)
            if res["constraint_score"] < 1.0 and test.constraints:
                if res["failure_type"] == "UNKNOWN_FAILURE": # Don't overwrite retrieval failure
                    res["failure_type"] = "CONSTRAINT_FAILURE"

        # Check for pipeline exception
        if output.get("error") and not test.is_negative:
             res["failure_type"] = "PIPELINE_EXCEPTION"

        # Final Status
        if res["routing_ok"]:
            if test.is_negative:
                res["status"] = "PASSED"
                res["failure_type"] = None
            elif res["ground_truth_found"] and res["constraint_score"] >= 0.5:
                res["status"] = "PASSED"
                res["failure_type"] = None

        return res

    def run_evaluation(self):
        print(f"\n{'='*70}")
        print(f"STARTING UPGRADED E2E EVALUATION ({len(self.test_cases)} tests)")
        print(f"{'='*70}\n")

        for test in self.test_cases:
            print(f"Test {test.id}: {test.name}...", end=" ", flush=True)
            start_time = time.time()
            
            try:
                # Call the synchronous edge of the application
                output = run_pipeline_interface(test.prompt)
                duration = time.time() - start_time
                
                # Rate-limit to prevent overloading LLM provider
                time.sleep(1)
                
                eval_metrics = self._evaluate_results(test, output)
                self.results.append({
                    "test": test,
                    "metrics": eval_metrics,
                    "duration": duration,
                    "error": output.get("error")
                })
                
                print(f"[{eval_metrics['status']}] ({duration:.2f}s)")
            except Exception as e:
                print(f"[EXCEPTION] {str(e)}")
                self.results.append({
                    "test": test,
                    "metrics": {"status": "ERROR", "failure_type": "PIPELINE_EXCEPTION"},
                    "duration": 0,
                    "error": str(e)
                })

        self.print_report()

    def print_report(self):
        print(f"\n{'='*70}")
        print(f"{' '*25}E2E EVALUATION REPORT")
        print(f"{'='*70}\n")
        
        total = len(self.test_cases)
        passed = sum(1 for r in self.results if r["metrics"].get("status") == "PASSED")
        
        # Aggregate Metrics
        mrr = sum(r["metrics"].get("rr", 0) for r in self.results) / total
        r1 = sum(r["metrics"].get("recall_at_1", 0) for r in self.results) / total
        r5 = sum(r["metrics"].get("recall_at_5", 0) for r in self.results) / total
        avg_cons = sum(r["metrics"].get("constraint_score", 0) for r in self.results) / total
        
        # Taxonomy
        tax = {}
        for r in self.results:
            ft = r["metrics"].get("failure_type")
            if ft:
                tax[ft] = tax.get(ft, 0) + 1

        # Confidence
        conf_stats = {"routing": [], "retrieval": []}
        overconfident_failures = 0
        for r in self.results:
            c = r["metrics"].get("confidences", {})
            if c.get("routing") is not None: conf_stats["routing"].append(c["routing"])
            if c.get("retrieval") is not None: conf_stats["retrieval"].append(c["retrieval"])
            
            # Check overconfidence
            max_conf = max([v for v in c.values() if v is not None] or [0])
            if max_conf > 0.8 and r["metrics"]["status"] == "FAILED":
                overconfident_failures += 1

        # Length Stats
        lengths = [len(t.prompt) for t in self.test_cases]

        print(f"OVERALL SCORE: {passed}/{total} ({passed/total:.1%})")
        print(f"Latency: Avg {sum(r['duration'] for r in self.results)/total:.2f}s | Range {min(r['duration'] for r in self.results):.2f}-{max(r['duration'] for r in self.results):.2f}s")
        print(f"Query Length: Avg {sum(lengths)/total:.0f} | Min {min(lengths)} | Max {max(lengths)}")

        print(f"\n[RANKING METRICS]")
        print(f"MRR:         {mrr:.3f}")
        print(f"Recall@1:    {r1:.3f}")
        print(f"Recall@5:    {r5:.3f}")

        print(f"\n[CONSTRAINT ANALYSIS]")
        print(f"Avg Constraint Score: {avg_cons:.2%}")

        print(f"\n[FAILURE TAXONOMY]")
        if not tax:
            print("No failures recorded.")
        for k, v in tax.items():
            print(f"{k:<20}: {v}")

        print(f"\n[CONFIDENCE ANALYSIS]")
        avg_route = sum(conf_stats["routing"])/len(conf_stats["routing"]) if conf_stats["routing"] else 0
        avg_retr = sum(conf_stats["retrieval"])/len(conf_stats["retrieval"]) if conf_stats["retrieval"] else 0
        print(f"Avg Routing Conf:   {avg_route:.2%}")
        print(f"Avg Retrieval Conf: {avg_retr:.2%}")
        print(f"Overconfident Failures: {overconfident_failures}")

        print(f"\n{'ID':<4} | {'Test Name':<35} | {'Status':<8} | {'RR':<5} | {'Cons'}")
        print("-" * 70)
        for res in self.results:
            t = res["test"]
            m = res["metrics"]
            print(f"{t.id:<4} | {t.name:<35} | {m['status']:<8} | {m.get('rr', 0):.2f} | {m.get('constraint_score', 0):.2f}")

        print(f"\n{'='*70}")

if __name__ == "__main__":
    evaluator = BioSeqEvaluator()
    evaluator.run_evaluation()
