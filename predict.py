import os
import json
import configparser
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from transformers.data.processors.squad import SquadExample

from arguments import MODEL_CLASS_TABLE, get_parser
from processors.preprocess import squad_convert_examples_to_features
from processors.postprocess import compute_predictions_logits
from eval_func import eval_by_model
from models.derived_models import BioModelClassify

QUES_ID = 0
CONCRETE_QUESTION = {
    'class': "Whether XX is gram-positive or gram-negative?", 
    'part': "Where does XX normally exist?", 
    'disease': "What kinds of disease can XX cause?", 
    'pathogenicity': "What about the pathogenicity of XX?", 
    'sensitivity': "What kinds of drugs is XX sensitive to?", 
    'tolerance': "What kinds of drugs is XX resistant to?", 
    'aerobic': "How about XX's requirement for oxygen?", 
    'morphology': "What is the shape of XX?", 
}


class Prophet:

    def __init__(self, config_file_path) -> None:
        with open(config_file_path) as fp:
            jsonStr = fp.read()
        self._args = parse_json_to_obj(jsonStr, SquadArgs)
        self._threshold = 0.9
        
        config = AutoConfig.from_pretrained(
            self._args.config_name,
            cache_dir=self._args.cache_dir, 
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._args.tokenizer_name,
            do_lower_case=self._args.do_lower_case,
            cache_dir=self._args.cache_dir, 
            use_fast=False,
        )

        target_model = MODEL_CLASS_TABLE[self._args.model_class]
        self._model = target_model(self._args.model_name_or_path, config, self._args)
        self._model.to(self._args.device)
        self._model.eval()
            

    def __ask(self, examples) -> Dict:
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self._tokenizer,
            max_seq_length=self._args.max_seq_length,
            doc_stride=self._args.doc_stride,
            max_query_length=self._args.max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=self._args.threads,
            tqdm_enabled=self._args.tqdm_enabled,
        )

        eval_dataloader = DataLoader(dataset, batch_size=self._args.per_gpu_eval_batch_size)

        all_results, all_pred_unsolvable = eval_by_model(self._args, self._model, eval_dataloader, features, tqdm_enabled=self._args.tqdm_enabled)

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            self._args.n_best_size,
            self._args.max_answer_length,
            self._args.do_lower_case,
            None,
            None,
            None,
            self._args.verbose_logging,
            self._args.with_neg,
            self._args.null_score_diff_threshold,
            self._tokenizer,
            all_pred_unsolvable if isinstance(self._model, BioModelClassify) else None,
        )

        return predictions


    def get_abbr(self, name: str):
        idx = name.find(' ')
        return name.replace(name[1:idx], '.') if idx != -1 else None
    
    def make_example(self, ques, section, key: str=None, section_title: str=None):
        return SquadExample(
            qas_id=key if key else QUES_ID,
            question_text=ques,
            context_text=section,
            answer_text=None,
            start_position_character=None,
            title=section_title if section_title else "foo",
            is_impossible=False,
            answers=None,
        )

    def make_examples(self, ques, section, key: str=None, section_title: str=None):
        return [self.make_example(ques, section, key, section_title)]
    

    def ask_one_section(self, species_name, section):
        all_ans = {}

        for key in CONCRETE_QUESTION:
            ques = CONCRETE_QUESTION[key].replace("XX", species_name)
            examples = self.make_examples(ques, section)
            ans = self.__ask(examples)[QUES_ID]

            if ans["offset"] == -1:
                abbr_name = self.get_abbr(species_name)
                ques = CONCRETE_QUESTION[key].replace("XX", abbr_name)
                examples = self.make_examples(ques, section)
                ans = self.__ask(examples)[QUES_ID]
            
            if ans and ans["text"] and ans["prob"] > self._threshold:
                all_ans[key] = {"text": ans["text"], "char_start": ans["offset"]}

        return all_ans


    def ask_serial(self, species_name, json_content):
        content = json.loads(json_content)

        ans_table = {}
        for key in content:
            if content[key] != None:
                ans = self.ask_one_section(species_name, content[key])
                for ques in ans:
                    if ques not in ans_table:
                        ans_table[ques] = defaultdict(list)
                    ans_table[ques][ans[ques]["text"]].append({"section": key, "char_start": ans[ques]["char_start"]})

        return ans_table


    def ask(self, species_name, json_content):
        content = json.loads(json_content)

        ans_table = {}
        idx = 0        
        examples = []

        for ques_type in CONCRETE_QUESTION:
            ans_table[ques_type] = None

        for part in content:
            if content[part] != None:
                section = content[part]
                for key in CONCRETE_QUESTION:
                    ques = CONCRETE_QUESTION[key].replace("XX", species_name)
                    example = self.make_example(ques, section, key=key+f"_{idx}", section_title=part)
                    examples.append(example)
                    idx += 1

        predictions = self.__ask(examples)
        for key in predictions:
            ques_type = key.split("_")[0]
            ans = predictions[key]
            if ans and ans["text"] and ans["prob"] > self._threshold:
                if ans_table[ques_type]:
                    if ans_table[ques_type]["prob"] > ans["prob"]:
                        ans_table[ques_type] = ans
                else:
                    ans_table[ques_type] = ans

        if self._args.with_neg:
            examples.clear()
            abbr_name = self.get_abbr(species_name)
            for ques_type in ans_table:
                if ans_table[ques_type]:
                    for part in content:
                        if content[part] != None:
                            section = content[part]
                            ques = CONCRETE_QUESTION[ques_type].replace("XX", abbr_name)
                            example = self.make_example(ques, section, key=key+f"_{idx}", section_title=part)
                            examples.append(example)
                            idx += 1
            if examples:
                predictions = self.__ask(examples)
                for key in predictions:
                    ques_type = key.split("_")[0]
                    ans = predictions[key]
                    if ans and ans["text"] and ans["prob"] > self._threshold:
                        if ans_table[ques_type]:
                            if ans_table[ques_type]["prob"] > ans["prob"]:
                                ans_table[ques_type] = ans
                        else:
                            ans_table[ques_type] = ans

        ret_ans_table = {}
        for ques_type in CONCRETE_QUESTION:
            ret_ans_table[ques_type] = defaultdict(list)
        for ques_type in ans_table:
            if ans_table[ques_type]:
                ret_ans_table[ques_type][ans_table[ques_type]["text"]] = {"section": ans_table[ques_type]["title"], "char_start": ans_table[ques_type]["offset"]}

        return ret_ans_table


class SquadArgs:

    pass
    
def parse_json_to_obj(jsonStr, objClass):
    parseData = json.loads(jsonStr.strip('\t\r\n'))
 
    result = objClass()
    result.__dict__ = parseData
 
    return result

def custom_args_to_json(save_name: str, custom_args_file_path: str=""):
    args = get_parser()
    if os.path.exists(custom_args_file_path):
        custom_args = configparser.ConfigParser()
        custom_args.read(custom_args_file_path)

        custom_args_sec = custom_args["custom.integer"]
        for key in custom_args_sec:
            args.__dict__[key] = custom_args_sec.getint(key)
        custom_args_sec = custom_args["custom.float"]
        for key in custom_args_sec:
            args.__dict__[key] = custom_args_sec.getfloat(key)
        custom_args_sec = custom_args["custom.boolean"]
        for key in custom_args_sec:
            args.__dict__[key] = custom_args_sec.getboolean(key)
        custom_args_sec = custom_args["custom.string"]
        for key in custom_args_sec:
            args.__dict__[key] = custom_args_sec[key]
    with open(save_name, 'wt') as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":

    import timeit
    # species_name = "Bacillus coagulans"
    # content = {
    #     "abstract" : "Probiotic microorganisms are generally considered to beneficially affect host health when used in adequate amounts. Although generally used in dairy products, they are also widely used in various commercial food products such as fermented meats, cereals, baby foods, fruit juices, and ice creams. Among lactic acid bacteria, Lactobacillus and Bifidobacterium are the most commonly used bacteria in probiotic foods, but they are not resistant to heat treatment. Probiotic food diversity is expected to be greater with the use of probiotics, which are resistant to heat treatment and gastrointestinal system conditions. Bacillus coagulans (B. coagulans) has recently attracted the attention of researchers and food manufacturers, as it exhibits characteristics of both the Bacillus and Lactobacillus genera. B. coagulans is a spore-forming bacterium which is resistant to high temperatures with its probiotic activity. In addition, a large number of studies have been carried out on the low-cost microbial production of industrially valuable products such as lactic acid and various enzymes of B. coagulans which have been used in food production. In this review, the importance of B. coagulans in food industry is discussed. Moreover, some studies on B. coagulans products and the use of B. coagulans as a probiotic in food products are summarized. ",
    #     "introduction" : "Nowadays, the interest in probiotic foods is increasing due to the growing consumer demand for safe and functional foods with health-promoting properties and high nutritional value [1]. Probiotics are defined as “live microorganisms that, when administered in adequate amounts, confer a health benefit on the host” [2]. In order to obtain benefits, probiotic products should contain at least 10–10 cfu/g probiotic microorganism and should survive until the end of shelf life [3]. Probiotic microorganisms, which are naturally found in intestinal microbiota, could protect humans from diseases, modulate and strengthen the immune system, prevent tooth decay, have anticarcinogenic properties, and be effective against coronary heart disease [4,5]. Probiotic microorganisms can produce organic acids (such as lactic and acetic acid), hydrogen peroxide, and bacteriocin [5]. Probiotics have several mechanisms to inhibit pathogen microorganisms. The primary mechanisms are as follows: (1) the lowering of the pH of food through lactic acid production; (2) the production of antimicrobial substances such as microcin, hydrogen peroxide, and compounds like free radicals; (3) competition for food resources by attaching to receptors; and (4) stimulation of the production of secretory IgA (Immunoglobulin A) by the formation of protective mucin (parent substance of the mucus composed of tissue of epithelial or connective origin and a mixture of glycoprotein and mucoprotein) [5].  There are two basic forms of probiotic microorganisms used in foods: the vegetative form and the spore form. The vegetative form is more susceptible to high temperatures, moisture, acidity, shelf life of food, and negative environmental conditions during the manufacture of food than the spore form. However, some probiotic microorganisms do not have spore forms [4]. Fermentation conditions, freezing, thawing, drying, cell protection additives, rehydration of dried probiotics, and microencapsulation applications are factors that affect the survival of probiotic microorganisms during probiotic food production. Food compounds, food additives, oxygen content, redox potential, moisture content/water activity, storage temperature, pH and titration acidity, and packaging conditions are factors that also affect survival of probiotic microorganisms during storage [6]. Gastrointestinal system conditions and stress factors could cause significant loss of viable probiotic cells [7].  Lactic acid bacteria (LAB; for example, Lactobacillus and Bifidobacterium and some Saccharomyces species) are the microorganisms most commonly used in probiotic food production [8,9,10,11]. However, these microorganisms cannot survive heat treatment, for which the cold spot temperature is approximately 75 °C [8,10]. Heat treatment is not applicable for most probiotic foods that contain commercial probiotic microorganisms due to their sensitivity to heat. Nevertheless, it has been stated that this restriction could be overcome by the usage of spore-forming probiotic microorganisms. It is known that some non-pathogenic Bacillus species, which are not as well-known as LAB and yeasts, are being used as probiotics [12]. The survival and stability of these bacteria have considerably improved compared to others through their spore-forming abilities. They are identified as an ideal choice in order to development of functional foods by protecting their vitality in high-temperature applications [13,14]. Bacillus coagulans (B. coagulans) was firstly isolated from spoiled milk [6]. In 1933, it was identified as Lactobacillus sporogenes by Horowitz-Wlassowa and Nowotelnow. Afterwards, it was classified as B. coagulans [15]. B. coagulans is a gram-positive, facultative anaerobic, nonpathogenic, spore-forming, lactic acid-producing bacteria [4]. It is resistant to heat; the optimum growth temperature for B. coagulans is 35 to 50 °C and the optimum growth pH is 5.5 to 6.5 [4,15]. It has the characteristics of microorganisms used as probiotics [15]. Some strains of B. coagulans have been reported as facultative anaerobe, thermophile bacteria able to grow at pH 6.2, 60–65 °C [6,16]. Although B. coagulans produces acid, it does not produce gas from maltose, raffinose, mannitol, and sucrose fermentation. It was reported that B. coagulans causes deterioration in dairy, fruit, and vegetable products due to acid production. In addition to lactic acid production, some strains also produce thermostable α-amylase [4,17]. For this reason, B. coagulans is important from an industrial point of view. B. coagulans spores are terminal, while spores of other species are central or subterminal. Furthermore, it differs from other Bacillus species due to the absence of cytochrome-C oxidase, and it does not reduce nitrate to nitrite [4]. It was reported that B. coagulans could grow at pH 4.5 at 65 °C and was isolated from products containing milk and carbohydrate [18]. B. coagulans has been reported as safe by the US Food and Drug Administration (FDA) and the European Union Food Safety Authority (EFSA) and is on the Generally Recognized As Safe (GRAS) and Qualified Presumption of Safety (QPS) list [19]. In addition, it was reported that genome sequencing can provide information about the overall characterization of the bacterium, for example with respect to its safety as a food supplement [20]. The B. coagulans GBI-30, 6086 genome was investigated, and it was found that it did not contain any hazardous genes [21]. Some of the non-pathogenic strains among the 100 known Bacillus spp., including B. coagulans and Bacillus subtilis var. natto, were stated as safe for human consumption [22,23]. ",
    #     "discussion" : None,
    #     "conclusion" : "Consumer interest in healthier and more functional food is increasing due to changing consumption habits and increasing interest in food and health. In addition to supporting the clinically beneficial effects of probiotic microorganisms on health, the formulation of probiotic food products has great importance for consumers, industry, and research centers which are interested in the subject. Heat-resistance of probiotic Bacillus spp. spore forms can provide an advantage for heat-treated probiotic foods. B. coagulans is attracting interest due to its resistance to strong gastric acid and high temperatures, and it is more resistant to antibiotics than other LAB. Moreover, the products, which are used in food industry and could be produced by B. coagulans, are gaining attention due to their low cost and as an alternative to other chemical sources. ",
    # }
    species_name = "Rothia mucilaginosa"
    content = {
        "abstract" : "Rothia mucilaginosa is a gram-positive coccus of the family Micrococcaceae . R. mucilaginosa is considered a part of the normal flora of the human oropharynx and upper respiratory tract and lower respiratory tract infections attributable to R. mucilaginosa are not frequent. We present a case of pneumonia, in which the R. mucilaginosa infection was diagnosed by quantitative cultures of a bronchoalveolar lavage (BAL) specimen. A 46-yr-old woman with B lymphoblastic lymphoma was admitted to the hospital for scheduled chemotherapy. Her chest computed tomography (CT) scan revealed bilateral multifocal nodular and patchy consolidation in both lungs. Investigation of the BAL specimen revealed that 7% of leukocytes had intracellular gram-positive cocci . The quantitative cultures of the BAL specimen grew mucoid, non-hemolytic, and grayish convex colonies on blood agar at a count of approximately 200,000 colony-forming units/mL. The colonies were identified as R. mucilaginosa. The patient was empirically treated with levofloxacin for 7 days, after which findings on the chest radiograph and CT scan improved. She was discharged with improvement on hospital day 46. To our knowledge, this is the first report of R. mucilaginosa pneumonia diagnosed in Korea. Quantitative culture of BAL specimen and examination of intracellular organisms are crucial for assessing the clinical significance of R. mucilaginosa recovered from the lower respiratory tract.",
        "introduction" : "Rothia mucilaginosa, previously Stomatococcus mucilaginosus, is an aerobic, gram-positive coccus belonging to the family Micrococcaceae . It is found in the oropharynx and upper respiratory tract as part of the normal flora [1] and was first isolated from an endocarditis patient in 1978 [2]. It is considered an opportunistic pathogen, most often seen in immunocompromised patients, but it is also (less frequently) observed in immunocompetent subjects [3]. There are reports of R. mucilaginosa as a cause of bacteremia, central nervous system infection, meningitis, peritonitis, osteomyelitis, cervical necrotizing fasciitis, endophthalmitis, and endocarditis [1, 2, 4, 5]. Only a dozen cases of lower respiratory tract infection caused by R. mucilaginosa have been diagnosed by recovery of R. mucilaginosa from the bronchoscopic, blood, or sputum specimens [6-9]. R. mucilaginosa is generally considered to be a contaminant in respiratory tract infection specimens. Therefore, for R. mucilaginosa to be confirmed as a cause of a lower respiratory tract infection, the diagnostic specimen must be minimally contaminated. In this study, we report a case of pneumonia due to R. mucilaginosa that was diagnosed by quantitative cultures and visualization of intracellular organisms from a bronchoalveolar lavage (BAL) specimen from a lymphoma patient.",
        "discussion" : None,
        "conclusion" : "In this study, we present a case of pneumonia, in which R. mucilaginosa was the only significant organism recovered from the BAL culture. This intracellular organism was considered to be the causative pathogen of the pneumonia by combined analysis of the quantitative culture and visualization of ICOs. The patient' s chest radiograph and CT scan indicated likely atypical pneumonia, and the BAL specimen was also positive for rhinovirus. Therefore rhinovirus could not be ruled out as a causative agent of pneumonia. However, considering the purulent BAL specimen, we believed that the bacteria played at least a partially pathogenic role, and the patient was treated with antimicrobial therapy. The predominant growth of R. mucilaginosa in the BAL specimens with rare epithelial cells was not likely due to contamination by the normal oropharyngeal flora. There have been fewer than 20 cases of lower respiratory tract infection caused by R. mucilaginosa reported worldwide [9], and the clinical manifestation of disease have ranged from mild bronchitis to pneumonia or recurrent lung abscesses [6]. However, it is difficult to determine the clinical significance of R. mucilaginosa when it is found in respiratory specimens. The diagnosis of pneumonia caused by R. mucilaginosa requires at least a culture from the bronchoscopic specimen [6, 9]. In this case, the quantitative cultures from the BAL specimen provided more than 10 CFU\/mL, which was enough to diagnose the causative agents of pneumonia [11]. These culture findings supported R. mucilaginosa as a true pathogen. In this case, analysis of the direct smear revealed significant numbers of macrophages and neutrophils engulfing gram-positive cocci in clusters as ICOs. The quantitative cultures of the BAL specimen grew organisms with the same morphology as these ICOs. Detecting ICOs in the BAL fluid is an early indication of an infectious pulmonary process [12, 13], and a cytologic study of the BAL specimen to identify ICOs would allow us to determine the causative agents of nosocomial pneumonia [14]. Quantitative cultures of the BAL specimens show moderate correlation with the percentage of ICOs in the specimen, which in turn correlates with the lung bacterial burden [13]. The percentage of neutrophils that contain ICOs is higher in patients with pneumonia than in those without pneumonia [14]; the cut-off varies from 1% to more than 20% [14]. The Centers for Disease Control and Prevention\/National Healthcare Safety Network recommends 5% as a critical value for pneumonia diagnosis [15]. There are no previous reports of pneumonia due to R. mucilaginosa that was diagnosed by quantitative analysis of BAL cultures and ICOs. We identified R. mucilaginosa using the API Staph Identification Panel (bioM\u00e9rieux SA). Several other commercial kits are available for the identification of R. mucilaginosa; however, their accuracy for this rarely isolated species has been questioned [12, 16]. Because R. mucilaginosa is not a common pathogen in respiratory specimens, we confirmed the identification by 16S rRNA gene sequencing. The morphology of this organism was consistent with its identification as R. mucilaginosa, which is an encapsulated gram-positive coccus found in pairs, clusters, and tetrads, with colonies that are mucoid, rubbery, or sticky in consistency and adherent to agar [1, 16, 17]. The isolate was resistant to gentamicin, clindamycin, and levofloxacin. R. mucilaginosa has a variety of antimicrobial susceptibilities [6, 17-19]. Although the interpretation of antimicrobial susceptibility testing is not standardized, third-generation cephalosporins, vancomycin, high-dose ampicillin, rifampin, and chloramphenicol are consistently active against this bacterium [4, 17]. R. mucilaginosa has a broad range of susceptibility to other agents such as penicillin, clindamycin, and erythromycin [17]. Therefore, the susceptibility pattern of this isolate was generally consistent with that of R. mucilaginosa from previous reports [20, 21]. When the MIC breakpoints for Corynebacterium spp. were applied in this case [10], the organism was susceptible to penicillin, clindamycin, erythromycin, ceftriaxone, cefotaxime, cefepime, meropenem, trimethoprim\/sulfamethoxazole, tetracycline, vancomycin, and imipenem. The other antimicrobials were interpreted according to the MIC breakpoints for Staphylococcus spp. [22]; the pathogen was susceptible to ampicillin, amoxicillin\/clavulanate, azithromycin, chloramphenicol, cefuroxime, and piperacillin\/tazobactam, and resistant to levofloxacin. We can speculate on the emergence of acquired resistance to fluoroquinolones because the patient had a history of ciprofloxacin treatment. There are previous reports of emerging resistance of R. mucilaginosa to ciprofloxacin [18] and trimethoprim\/sulfamethoxazole [5] in association with prophylactic antimicrobial therapy. The variable susceptibility of R. mucilaginosa to \u03b2-lactams, aminoglycosides, macrolides, and fluoroquinolones dictates that the choice of antimicrobial agents should be guided by individual susceptibility tests in cases of severe infection. Rifampin, penicillin, ciprofloxacin, gentamicin, and clarithromycin have been previously used to treat R. mucilaginosa pneumonia [6]. Although in our case the isolate was resistant to levofloxacin, the pneumonia improved after levofloxacin therapy. Levofloxacin can be concentrated in respiratory tissues and intracellular compartments [23, 24] and is the treatment of choice for community-acquired pneumonia. In our case, levofloxacin may have been more active than in vitro susceptibility results on ICOs would have led us to expect. Outcomes of R. mucilaginosa pneumonia treated with levofloxacin are generally favorable, but non-attributable mortality has been found in patients with serious underlying disease [6, 7, 9]. In conclusion, this is the first description of pneumonia caused by R. mucilaginosa in Korea. This case indicates that quantitative analysis of BAL culture and ICOs are useful for diagnosing pneumonia caused by R. mucilaginosa.",
    }
    # species_name = "Rothia mucilaginosa"
    # content = {
    #     "abstract" : "Rothia mucilaginosa is increasingly recognized as an emerging opportunistic pathogen associated with prosthetic device infections. Infective endocarditis is one of the most common clinical presentations. We report a case of R. mucilaginosa prosthetic valve endocarditis and review the literature of prosthetic device infections caused by this organism.",
    #     "introduction" : "A 36-year-old man was admitted to the hospital in January 2012 with a chief complaint of left foot pain for 1 week. He described redness and swelling on the dorsum of his left foot. He denied trauma to the foot. He had been taking acetaminophen for pain intermittently without relief. He denied fever or chills, visual changes, back pain, muscle weakness, or numbness. He had a history of Streptococcus mitis mitral valve endocarditis and required mechanical mitral valve replacement in 2009. He had no history of peripheral vascular disease or claudication. He had no known drug allergies. His home medications included warfarin, methadone, and acetaminophen. He was an active intravenous heroin user. He was a former tobacco user with a 5-pack-year history who had quit 7 years before. On examination, the patient appeared well. His temperature was 100.9\u00b0F (38.3\u00b0C), pulse 108 beats per minute, blood pressure 133\/64 mm Hg, and respirations 20 per minute. Cardiovascular examination revealed normal S1 and S2 and no murmurs, rubs, or gallops. The dorsum of the left foot had mild erythema, slight edema, and point tenderness of the mid-dorsal region. The left dorsalis pedis pulse was easily palpable. Skin examination revealed track marks at the right antecubital fossa. The remainder of the examination was normal. Laboratory studies revealed a white blood cell count of 20 \u00d7 10 cells\/mm (reference range, 4 \u00d7 10 to 11 \u00d7 10\/mm), neutrophils at 88%, creatinine at 0.8 mg\/dl, and an erythrocyte sedimentation rate of 35 mm\/h (reference range, 0 to 10 mm\/h). Other routine laboratory tests were normal. A left-foot radiograph revealed no fracture, and an ultrasound of the left lower extremity revealed no deep vein thrombosis. Intravenous vancomycin and piperacillin-tazobactam were administered empirically for a presumptive diagnosis of left-foot cellulitis. The fever resolved, but the patient had persistent pain in the left foot, which subsequently turned blue and felt cold. Computed tomographic angiography revealed left popliteal artery thrombosis. A left popliteal thromboembolectomy was performed on day 4 of hospitalization. The pathology of the left popliteal thrombus revealed an organized thrombus with clusters of Gram-positive cocci. On day 4 of hospitalization, two sets of blood cultures obtained on the day of admission grew Rothia mucilaginosa from the aerobic bottles only. The organism was identified based upon biochemical tests, automated identification platforms (Phoenix system), and phenotypic characteristics. Gram stain revealed Gram-positive cocci that were catalase negative and grew sticky \u201cstaph-like\u201d colonies which were whitish to gray in color, nonhemolytic, smooth, and round (Fig. 1). Remel Bacticard strep reactions revealed positive results for l-leucine-beta-naphthylamide (LAP), l-pyroglutamyl-beta-naphthylamide (PYR), and esculin and ferric citrate (ESC). The BD Phoenix automated microbiology system for identification and antimicrobial susceptibility testing was used and revealed a 99% confidence value for Rothia mucilaginosa identification, with a profile number of 000003B284506FC6. Additional identification methods were not deemed necessary. (A) Growth on chocolate agar produced sticky colonies that adhered to the agar. (B) Growth on sheep blood agar produced sticky, \u201cstaph-like\u201d colonies that were whitish to gray in color, nonhemolytic, smooth, and round and exhibited the typical tenacious morphology of Rothia mucilaginosa. Transthoracic and transesophageal echocardiography revealed several mobile echodensities on the mechanical mitral valve prosthesis, with valve dehiscence, multiple areas of perforation, and paravalvular regurgitation. A diagnosis of R. mucilaginosa prosthetic valve endocarditis was made. The patient underwent mitral valve replacement on hospital day 14. The postoperative course was uneventful. The left foot appeared erythematous and warm, with less edema and tenderness. The patient was discharged to a subacute care facility on hospital day 26, with plans to complete a 6-week-total postoperative course of intravenous vancomycin (antimicrobial susceptibility results were unavailable at time of discharge). He was seen in the emergency department for an unrelated issue 1 week after antibiotic completion. During that visit, he was afebrile and had no clinical signs or symptoms of active infection. He was lost to follow-up thereafter. Subsequently, antimicrobial susceptibilities were reported. Susceptibility testing was performed using the AB Biodisk Etest. The MIC results demonstrated sensitivity to penicillin (0.016 \u03bcg\/ml), vancomycin (2.0 \u03bcg\/ml), ceftriaxone (0.016 \u03bcg\/ml), and daptomycin (2.0 \u03bcg\/ml). Rifampin testing was performed by the Kirby Bauer method using a BD BBL 5-\u03bcg rifampin disk. The zone size was 35 mm. Rothia mucilaginosa was formerly known as Staphylococcus salivarius, Micrococcus mucilaginosus, and Stomatococcus mucilaginosus. It was reclassified into a new genus belonging to the family Micrococcaceae in 2000 (1) based on 16S rRNA sequencing. The organism is an oxidase-negative, catalase-variable Gram-positive coccus bacterium. Gram staining reveals non-spore-forming, encapsulated Gram-positive cocci that can appear in pairs, tetrads, or irregular clusters. It is a facultative anaerobic bacterium which grows well on most nonselective media and in standard blood culture systems. On sheep blood and chocolate agar, the bacterium forms clear to gray\/white, nonhemolytic, mucoid or sticky colonies that adhere to the agar surface. It can be difficult to distinguish from coagulase-negative staphylococci, micrococci, and streptococci based on the catalase test result. Its inability to grow in 6.5% sodium chloride and its ability to hydrolyze gelatin and esculin distinguish it from species of the Staphylococcus, Micrococcus, and Enterococcus genera (2). Identification from automatic methods should correlate with phenotypic identification; otherwise, genetic sequencing may be needed to identify this organism. R. mucilaginosa is a normal inhabitant of the human oral cavity and respiratory tract (2). It is an infrequent pathogen, mostly affecting immunocompromised hosts, such as patients with cancer and severe neutropenia, human immunodeficiency virus infection, alcoholism, diabetes mellitus, and chronic liver disease (3\u20135). Recently, infections in immunocompetent hosts have been reported with increasing frequency. Risk factors for this infection include intravenous drug abuse, cardiac valve disease, and the presence of prosthetic devices, especially prosthetic heart valves. Infections caused by this organism have been described in various organ systems, including patients with bacteremia (6), endovascular infection, (5, 7\u20139), central nervous system infection (10), ocular infection (11), bone and joint infection (12, 13), pulmonary infection (14), biliary tract infection (15), and skin and soft tissue infections (16). Endocarditis is by far the most commonly reported clinical manifestation caused by this organism. P\u00e9rez-Vega et al. reported a case series of R. mucilaginosa infective endocarditis in the literature. In this series, the typical patient with R. mucilaginosa endocarditis was a healthy patient with underlying cardiac disease or an intravenous drug abuser with prosthetic heart valves or mitral valve prolapse. In all patients, endocarditis affected the left-side valves, involving native valves and prosthetic valves almost equally. All patients with native-valve endocarditis recovered with antibiotic therapy alone. However, most patients with infected prosthetic heart valves required a combination of antibiotic therapy and surgical valve replacement (5). Table 1 describes 8 patients reported in the literature with prosthetic device infections caused by R. mucilaginosa (7\u20139, 17\u201319). The reported prosthetic devices include prosthetic heart valves (5 patients), a prosthetic hip (1 patient), a cerebral ventricle catheter (1 patient), and a peritoneal dialysis catheter (1 patient). Three of 5 (60%) patients with prosthetic valve endocarditis developed septic emboli. Six of 8 (75%) patients had a good outcome with antibiotic therapy combined with prosthetic device removal. Eight cases of Rothia mucilaginosa prosthetic device infections reported in the literature and our case Two of the 8 patients (25%) with prosthetic device infections died. Of the two patients who died, one had bioprosthetic mitral valve endocarditis complicated by septic emboli to the brain while receiving vancomycin, gentamicin, and rifampin therapy. Surgical valve replacement was considered; however, the patient had a cardiorespiratory arrest and expired on day 6 of hospitalization (8). The other death occurred in a patient who had aortic and mitral bioprosthetic valve endocarditis complicated by periaortic abscess and septic emboli to the brain. The patient deferred surgical valve replacement and expired after an 8-week course of vancomycin therapy (18). The pathogenesis of this organism in prosthetic device infection has not been well described. The organism's ability to produce a biofilm, similar to other Gram-positive bacteria, is believed to be a key pathogenic mechanism (20). The physical protective layer provided by the biofilm presumably facilitates adhesion of the organisms to devices and renders them relatively refractory to medical therapy. This biofilm likely causes local damage, such as disruption of prosthetic heart valves or loosening of implanted devices, or systemic manifestations, such as septic emboli. Antibiotic therapy alone is usually ineffective without surgical removal of the infected prosthetic device (21, 22). The patient we describe above illustrates the positive outcome utilizing a combination of antibiotic and surgical therapy. The optimal antimicrobial treatment of R. mucilaginosa infection has not been determined. The organism is generally susceptible to penicillin, ampicillin, cefotaxime, imipenem, rifampin, and vancomycin. It is frequently resistant to clindamycin and aminoglycosides, as well as to trimethoprim-sulfamethoxazole and ciprofloxacin (13). Daptomycin has in vitro activity against this organism (23). Partial resistance to penicillin has been reported in the literature (7, 19). Therefore, vancomycin is recommended as empirical therapy while awaiting susceptibility testing. In summary, R. mucilaginosa is increasingly recognized as an emerging opportunistic pathogen associated with prosthetic device infections. It may be difficult to identify and can easily be mistaken for staphylococci or streptococci. When this organism causes clinical infection, prosthetic valve endocarditis is not uncommon. A combination of antibiotic therapy and prompt removal of the infected device is probably necessary for a successful outcome. Physicians should be aware of this organism when treating patients infected with Gram-positive bacteria associated with prosthetic devices.",
    #     "discussion" : None,
    #     "conclusion" : "A 36-year-old man was admitted to the hospital in January 2012 with a chief complaint of left foot pain for 1 week. He described redness and swelling on the dorsum of his left foot. He denied trauma to the foot. He had been taking acetaminophen for pain intermittently without relief. He denied fever or chills, visual changes, back pain, muscle weakness, or numbness. He had a history of Streptococcus mitis mitral valve endocarditis and required mechanical mitral valve replacement in 2009. He had no history of peripheral vascular disease or claudication. He had no known drug allergies. His home medications included warfarin, methadone, and acetaminophen. He was an active intravenous heroin user. He was a former tobacco user with a 5-pack-year history who had quit 7 years before. On examination, the patient appeared well. His temperature was 100.9\u00b0F (38.3\u00b0C), pulse 108 beats per minute, blood pressure 133\/64 mm Hg, and respirations 20 per minute. Cardiovascular examination revealed normal S1 and S2 and no murmurs, rubs, or gallops. The dorsum of the left foot had mild erythema, slight edema, and point tenderness of the mid-dorsal region. The left dorsalis pedis pulse was easily palpable. Skin examination revealed track marks at the right antecubital fossa. The remainder of the examination was normal. Laboratory studies revealed a white blood cell count of 20 \u00d7 10 cells\/mm (reference range, 4 \u00d7 10 to 11 \u00d7 10\/mm), neutrophils at 88%, creatinine at 0.8 mg\/dl, and an erythrocyte sedimentation rate of 35 mm\/h (reference range, 0 to 10 mm\/h). Other routine laboratory tests were normal. A left-foot radiograph revealed no fracture, and an ultrasound of the left lower extremity revealed no deep vein thrombosis. Intravenous vancomycin and piperacillin-tazobactam were administered empirically for a presumptive diagnosis of left-foot cellulitis. The fever resolved, but the patient had persistent pain in the left foot, which subsequently turned blue and felt cold. Computed tomographic angiography revealed left popliteal artery thrombosis. A left popliteal thromboembolectomy was performed on day 4 of hospitalization. The pathology of the left popliteal thrombus revealed an organized thrombus with clusters of Gram-positive cocci. On day 4 of hospitalization, two sets of blood cultures obtained on the day of admission grew Rothia mucilaginosa from the aerobic bottles only. The organism was identified based upon biochemical tests, automated identification platforms (Phoenix system), and phenotypic characteristics. Gram stain revealed Gram-positive cocci that were catalase negative and grew sticky \u201cstaph-like\u201d colonies which were whitish to gray in color, nonhemolytic, smooth, and round (Fig. 1). Remel Bacticard strep reactions revealed positive results for l-leucine-beta-naphthylamide (LAP), l-pyroglutamyl-beta-naphthylamide (PYR), and esculin and ferric citrate (ESC). The BD Phoenix automated microbiology system for identification and antimicrobial susceptibility testing was used and revealed a 99% confidence value for Rothia mucilaginosa identification, with a profile number of 000003B284506FC6. Additional identification methods were not deemed necessary. (A) Growth on chocolate agar produced sticky colonies that adhered to the agar. (B) Growth on sheep blood agar produced sticky, \u201cstaph-like\u201d colonies that were whitish to gray in color, nonhemolytic, smooth, and round and exhibited the typical tenacious morphology of Rothia mucilaginosa. Transthoracic and transesophageal echocardiography revealed several mobile echodensities on the mechanical mitral valve prosthesis, with valve dehiscence, multiple areas of perforation, and paravalvular regurgitation. A diagnosis of R. mucilaginosa prosthetic valve endocarditis was made. The patient underwent mitral valve replacement on hospital day 14. The postoperative course was uneventful. The left foot appeared erythematous and warm, with less edema and tenderness. The patient was discharged to a subacute care facility on hospital day 26, with plans to complete a 6-week-total postoperative course of intravenous vancomycin (antimicrobial susceptibility results were unavailable at time of discharge). He was seen in the emergency department for an unrelated issue 1 week after antibiotic completion. During that visit, he was afebrile and had no clinical signs or symptoms of active infection. He was lost to follow-up thereafter. Subsequently, antimicrobial susceptibilities were reported. Susceptibility testing was performed using the AB Biodisk Etest. The MIC results demonstrated sensitivity to penicillin (0.016 \u03bcg\/ml), vancomycin (2.0 \u03bcg\/ml), ceftriaxone (0.016 \u03bcg\/ml), and daptomycin (2.0 \u03bcg\/ml). Rifampin testing was performed by the Kirby Bauer method using a BD BBL 5-\u03bcg rifampin disk. The zone size was 35 mm. Rothia mucilaginosa was formerly known as Staphylococcus salivarius, Micrococcus mucilaginosus, and Stomatococcus mucilaginosus. It was reclassified into a new genus belonging to the family Micrococcaceae in 2000 (1) based on 16S rRNA sequencing. The organism is an oxidase-negative, catalase-variable Gram-positive coccus bacterium. Gram staining reveals non-spore-forming, encapsulated Gram-positive cocci that can appear in pairs, tetrads, or irregular clusters. It is a facultative anaerobic bacterium which grows well on most nonselective media and in standard blood culture systems. On sheep blood and chocolate agar, the bacterium forms clear to gray\/white, nonhemolytic, mucoid or sticky colonies that adhere to the agar surface. It can be difficult to distinguish from coagulase-negative staphylococci, micrococci, and streptococci based on the catalase test result. Its inability to grow in 6.5% sodium chloride and its ability to hydrolyze gelatin and esculin distinguish it from species of the Staphylococcus, Micrococcus, and Enterococcus genera (2). Identification from automatic methods should correlate with phenotypic identification; otherwise, genetic sequencing may be needed to identify this organism. R. mucilaginosa is a normal inhabitant of the human oral cavity and respiratory tract (2). It is an infrequent pathogen, mostly affecting immunocompromised hosts, such as patients with cancer and severe neutropenia, human immunodeficiency virus infection, alcoholism, diabetes mellitus, and chronic liver disease (3\u20135). Recently, infections in immunocompetent hosts have been reported with increasing frequency. Risk factors for this infection include intravenous drug abuse, cardiac valve disease, and the presence of prosthetic devices, especially prosthetic heart valves. Infections caused by this organism have been described in various organ systems, including patients with bacteremia (6), endovascular infection, (5, 7\u20139), central nervous system infection (10), ocular infection (11), bone and joint infection (12, 13), pulmonary infection (14), biliary tract infection (15), and skin and soft tissue infections (16). Endocarditis is by far the most commonly reported clinical manifestation caused by this organism. P\u00e9rez-Vega et al. reported a case series of R. mucilaginosa infective endocarditis in the literature. In this series, the typical patient with R. mucilaginosa endocarditis was a healthy patient with underlying cardiac disease or an intravenous drug abuser with prosthetic heart valves or mitral valve prolapse. In all patients, endocarditis affected the left-side valves, involving native valves and prosthetic valves almost equally. All patients with native-valve endocarditis recovered with antibiotic therapy alone. However, most patients with infected prosthetic heart valves required a combination of antibiotic therapy and surgical valve replacement (5). Table 1 describes 8 patients reported in the literature with prosthetic device infections caused by R. mucilaginosa (7\u20139, 17\u201319). The reported prosthetic devices include prosthetic heart valves (5 patients), a prosthetic hip (1 patient), a cerebral ventricle catheter (1 patient), and a peritoneal dialysis catheter (1 patient). Three of 5 (60%) patients with prosthetic valve endocarditis developed septic emboli. Six of 8 (75%) patients had a good outcome with antibiotic therapy combined with prosthetic device removal. Eight cases of Rothia mucilaginosa prosthetic device infections reported in the literature and our case Two of the 8 patients (25%) with prosthetic device infections died. Of the two patients who died, one had bioprosthetic mitral valve endocarditis complicated by septic emboli to the brain while receiving vancomycin, gentamicin, and rifampin therapy. Surgical valve replacement was considered; however, the patient had a cardiorespiratory arrest and expired on day 6 of hospitalization (8). The other death occurred in a patient who had aortic and mitral bioprosthetic valve endocarditis complicated by periaortic abscess and septic emboli to the brain. The patient deferred surgical valve replacement and expired after an 8-week course of vancomycin therapy (18). The pathogenesis of this organism in prosthetic device infection has not been well described. The organism's ability to produce a biofilm, similar to other Gram-positive bacteria, is believed to be a key pathogenic mechanism (20). The physical protective layer provided by the biofilm presumably facilitates adhesion of the organisms to devices and renders them relatively refractory to medical therapy. This biofilm likely causes local damage, such as disruption of prosthetic heart valves or loosening of implanted devices, or systemic manifestations, such as septic emboli. Antibiotic therapy alone is usually ineffective without surgical removal of the infected prosthetic device (21, 22). The patient we describe above illustrates the positive outcome utilizing a combination of antibiotic and surgical therapy. The optimal antimicrobial treatment of R. mucilaginosa infection has not been determined. The organism is generally susceptible to penicillin, ampicillin, cefotaxime, imipenem, rifampin, and vancomycin. It is frequently resistant to clindamycin and aminoglycosides, as well as to trimethoprim-sulfamethoxazole and ciprofloxacin (13). Daptomycin has in vitro activity against this organism (23). Partial resistance to penicillin has been reported in the literature (7, 19). Therefore, vancomycin is recommended as empirical therapy while awaiting susceptibility testing. In summary, R. mucilaginosa is increasingly recognized as an emerging opportunistic pathogen associated with prosthetic device infections. It may be difficult to identify and can easily be mistaken for staphylococci or streptococci. When this organism causes clinical infection, prosthetic valve endocarditis is not uncommon. A combination of antibiotic therapy and prompt removal of the infected device is probably necessary for a successful outcome. Physicians should be aware of this organism when treating patients infected with Gram-positive bacteria associated with prosthetic devices.",
    # }

    # species_name = 'Nocardia thailandica'

    # content = {
    #     "abstract" : " Successful treatment for Nocardia thailandica is not well elucidated in the literature. To the best of our knowledge, N. thailandica has not yet been described in the medical literature to cause central nervous system (CNS) infection from brain abscess. We report the case of an immunocompromised patient who underwent successful treatment to treat his brain abscess caused by N. thailandica.   After failing medical therapy, the patient underwent a craniotomy, and tissue was sent for culture. Upon identification by 16S rDNA sequencing, the organism causing infection was identified to be N. thailandica.   Based on susceptibilities, the patient was treated with IV ceftriaxone 2 grams daily for five months. The patient demonstrated clinical and radiological improvement which persisted to 7 months after initiation of therapy.   To the best of our knowledge, this is the first documented case of a brain abscess due to N. thailandica which was successfully treated. Due to the location of the infection, ceftriaxone was chosen because of optimal CNS penetration. Ceftriaxone monotherapy demonstrated clinical and radiographic treatment success resulting in the successful treatment of this infection. ",
    #     "introduction" : " Nocardia spp. are responsible for both localized infections, such as pneumonia, and disseminated infections which can occur in the central nervous system (CNS). Immunosuppression is a known risk factor for nocardia infection. Nocardia thailandica is a rare species which has only been documented to cause infections in humans four times in the medical literature since its original classification in 2004. Here, we report the first documented case of a brain abscess due to N. thailandica which was successfully treated with ceftriaxone monotherapy. ",
    #     "discussion" : " Nocardia is a genus of the aerobic actinomycetes family, a large and diverse group of Gram-positive bacteria. Accurate and rapid identification of Nocardia spp. is critical to optimize empiric antimicrobial therapy. Routine identification of Nocardia to the species level is a time-consuming process. Furthermore, these phenotypic tests may be inconclusive and difficult to interpret, resulting in limitations in the identification of Nocardia species [1, 2]. Additionally, the genus Nocardia has undergone substantial taxonomic revisions with the advent of molecular methods, rendering interpretation of identification challenging compared to historic data [2]. Methods that do not rely upon differential growth characteristics including antibiotic profiles such as 16S rDNA sequencing and matrix-assisted laser desorption ionization-time of flight mass spectrometry (MALDI-TOF MS) can provide more rapid and accurate identifications of challenging organisms such as Nocardia spp [3].  N. thailandica was first identified in 2004 from a soft tissue infection. In that report, the authors did not provide information on antimicrobial therapy [4]. The next report of N. thailandica was by Reddy et al., who isolated twenty Nocardia spp. from ocular infections; N. thailandica represented just one of the twenty isolated species [5]. While this report gave more thorough information on the susceptibilities of the isolate, the patient's treatment regimen was not delineated. Canterino et al. reported a 66-year-old patient on immunosuppressive therapy status post-lung transplant who had pulmonary nocardiosis [6]. Upon identification of N. thailandica via percutaneous lung biopsy, the patient was treated with meropenem for one month, followed by oral minocycline to complete six to twelve months of therapy. Following six weeks of antibiotic therapy, follow-up imaging revealed a good overall response to therapy. Bourbour et al. reported a 53-year-old, immunocompetent man with chronic bronchitis, who presented with persistent fever and cough and was found to have nodular infiltrates on chest X-ray [7]. A bronchoalveolar lavage sample grew Nocardia thailandica and the patient was treated with TMP/SMX and linezolid for 6 months. The authors reported that the patient's symptoms resolved completely. Optimal therapy for Nocardia spp. has not been well established [1, 8]. Considerations for selecting therapy should be based on species of Nocardia identified, site, and severity of infection. Combination therapy against Nocardia spp. has been thought to provide enhanced activity and is recommended for initial treatment for most forms of nocardiosis. Single-drug therapy may be sufficient after species identification, and antimicrobial drug susceptibility information can be confirmed [8]. As no randomized controlled trials provide guidance on optimal treatment, this recommendation is largely based on clinical experience. Patients with CNS nocardiosis may have increased mortality; therefore, combination therapy is often strongly recommended [9]. Ceftriaxone, meropenem/imipenem, sulfonamides, linezolid, and amikacin are often options for the treatment of nocardiosis. However, based on drug susceptibility testing at the species level, there is a wide range of variation in coverage [10, 11].  Table 1 reveals antibiotic susceptibility data of Nocardia thailandica from the literature and our case. Susceptibility profile of Nocardia thailandica based on the existing published literature. AMK: amikacin; AMC: amoxicillin-clavulanic acid; AZI: azithromycin; FEP: cefepime; CRO: ceftriaxone; CIP: ciprofloxacin; CLR: clarithromycin; DOX: doxycycline; GAT: gatifloxacin; IPM, imipenem; LZD: linezolid; MIN: minocycline; MXF: moxifloxacin; TOB: tobramycin; SXT: trimethoprim-sulfamethoxazole. S: susceptible; I: intermediate; R: resistant; 1/2 strains were susceptible and the other resistant. Of the four strains of N. thailandica with full susceptibility data reported, all were susceptible to ceftriaxone and carbapenems. Of the most common susceptible agents, ceftriaxone and TMP/SMX have the most optimal blood-brain barrier penetration and may be ideal for the treatment of CNS nocardiosis [12]. Due to poor penetration into the CNS and an unfavorable toxicity profile with prolonged administration, amikacin would be a suboptimal option. Linezolid, despite good CNS penetration, has significant adverse effects associated with prolonged use, such as thrombocytopenia, peripheral neuropathy, and optic neuropathy, the latter two of which are irreversible. Imipenem also penetrates the CNS well; however, it has well-known toxicity of seizures limiting its use. Moxifloxacin reaches high CSF concentrations and is active against selected Nocardia spp.; however, it has extremely limited human data [10, 12]. Meropenem, while certainly an option for CNS nocardiosis, may prove to be overly broad and increase the risk of selecting resistant organisms. Additionally, the increased dosing frequency of meropenem would also prove to be a limitation of outpatient treatment as compared to the dosing regimen selected for our patient, once-daily dosing of ceftriaxone. In summary, ceftriaxone and TMP/SMX would be potentially ideal agents for the treatment of CNS nocardiosis given their coverage for Nocardia thailandia and good penetration into the CNS. ",
    #     "conclusion" : "This is the first documented case of a brain abscess due to N. thailandica which was successfully treated. Due to the location of the infection, ceftriaxone and TMP/SMX were chosen because of optimal CNS penetration. TMP/SMX was discontinued approximately three weeks after initiation due to hyperkalemia. Ceftriaxone monotherapy demonstrated clinical and radiographic treatment success resulting in the successful treatment of this infection. ",
    # }

    pt = Prophet(os.path.join(os.getcwd(), "config.json"))
    start_time = timeit.default_timer()
    rst = pt.ask(species_name, json.dumps(content))
    # rst = pt.ask_serial(species_name, json.dumps(content))
    evalTime = timeit.default_timer() - start_time
    print(f"Evaluation done in total {evalTime} secs")
    print(rst)
