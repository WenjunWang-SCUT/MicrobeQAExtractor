def produce_answers(ans_info, example, f1_score, output_file):

    content = example.context_text
    ans = ans_info['text']
    
    if ans:
        start = ans_info['offset']
        while start >= 0 and not content[start].startswith(" "):
            start -= 1
        start += 1

        end = start + len(ans) - 1
        if end >= len(content):
            end = len(content) - 1

        if start <= end:
            sentence_start = start
            sentence_end = end
            while sentence_start >= 0:
                if not content[sentence_start] == '.':
                    sentence_start -= 1
                else:
                    break
            while sentence_end < len(content) - 1:
                if not content[sentence_end] == '.':
                    sentence_end += 1
                else:
                    break

            content = (
                    "".join(content[:sentence_start + 1]) +
                    '<span style="background: yellow; color: black">' +
                    "".join(content[sentence_start + 1: start]) +
                    '<span style="background: blue; color: white">' +
                    "".join(content[start: end + 1]) +
                    "</span>" +
                    "".join(content[end + 1: sentence_end]) +
                    "</span>" +
                    "".join(content[sentence_end:])
            )
    else:
        start = -1
        end = -1
        
        pass

    if example.answer_text:
        ori_ans, ori_start = example.answer_text.split('#', 1)
    else:
        ori_ans = ""
        ori_start = -1

    output_file.write(f"""\n===========================================
    <table border="1">
        <tr>
        <td width="15%" valign="top"><b>标题</b></td>         <td>{example.title}</td> </tr>
        <tr>
        <td width="15%" valign="top"><b>上下文</b></td>       <td>{content}</td></tr>
        <tr>
        <td width="15%" valign="top"><b>问题</b></td>      <td>{example.question_text}</td></tr>
        <tr>
        <td width="15%" valign="top"><b>标记</b></td>         <td>{ori_ans}</td></tr>
        <tr>
        <td width="15%" valign="top"><b>标记位置</b></td>  <td>{ori_start}</td></tr>
        <tr>
        <td width="15%" valign="top"><b>预测</b></td>          <td>{ans}</td></tr>
        <tr>
        <td width="15%" valign="top"><b>预测位置</b></td>  <td>[{start}, {end + 1}]</td></tr>
        <tr>
        <td width="15%" valign="top"><b>F1得分</b></td>      <td>{f1_score}</td></tr>
    </table>
    """)


import re

ques_patterns = [
    "Whether [\s\S]* is gram-positive or gram-negative?",
    "Where does [\s\S]* normally exist?",
    "What kinds of diseases can [\s\S]* cause?",#disease->diseases
    "What about the pathogenicity of [\s\S]*?", 
    "How about the virulence of [\s\S]*?", 
    "What kinds of drugs are [\s\S]* sensitive to?", #is->are
    "What kinds of drugs are [\s\S]* resistant to?", #is->are
    "How about [\s\S]*'s requirement for oxygen?", 
    "Whether [\s\S]* has catalase?", 
    "What is the shape of [\s\S]*?", 
    "How about the motility of [\s\S]*?", 
    "Whether the [\s\S]* forms spores?",

    # augment
    # part
    "What are the typical habitats of [\s\S]*?",
    "In what environments can [\s\S]* be found?",
    "Where is [\s\S]* commonly present?",
    "In what locations can [\s\S]* usually be found?",
    "What are the common sites where [\s\S]* is known to inhabit?",

    # disease
    # "What kinds of diseases can [\s\S]* cause?",
    "What are the diseases that can be caused by [\s\S]*?",
    "Which diseases are associated with [\s\S]* infection?",
    "What health problems can result from [\s\S]* colonization?",
    "What is the disease spectrum of [\s\S]*?",
    "What types of illnesses can [\s\S]* contribute to?",

    # sensitivity
    "Which drugs are effective against [\s\S]*?",
    "What medications can be used to treat [\s\S]* infections?",
    "Which antibiotics are recommended for treating [\s\S]* infections?",
    "What drugs have been shown to be active against [\s\S]*?",
    "What are the drugs that [\s\S]* is vulnerable to?",

]

ques_types = [
    'class',
    'part',
    'disease',
    'pathogenicity',
    'virulence',
    'sensitivity',
    'tolerance',
    'aerobic',
    'catalase',
    'morphology',
    'moveability',
    'sporulation',

    # augment
    'part',
    'part',
    'part',
    'part',
    'part',

    # 'disease',
    'disease',
    'disease',
    'disease',
    'disease',
    'disease',

    'sensitivity',
    'sensitivity',
    'sensitivity',
    'sensitivity',
    'sensitivity',
]

def getQuesType(question):
    for idx, ques_pattern in enumerate(ques_patterns):
        res = re.match(ques_pattern, question)
        if res != None:
            return ques_types[idx]
    print(question)
    return None

if __name__ == '__main__':
    ques = 'What is the shape of C. curvus?'
    ret = getQuesType(ques)
    print(ret)
