import re
from lxml import etree
from dataclasses import dataclass
from definitions import FoundTerm
"""
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')

import textspan
"""

# Starting with
#  <p>(iii) <i>Determination of whether organization is publicly supported - In general.</i> ... </p>
# want
#  <p>(iii) <i>Determination of whether organization is <definition>publicly supported</definition> - In general.</i> ... </p>


def xml_texts(element):
    elems = element.xpath("//descendant-or-self::*")

    for elem in elems:
        print()
        print(f"{elem.tag=}")
        print(f"{elem.text=}")
        print(f"{elem.tail=}")
        print(etree.tostring(elem, method="text", encoding="unicode"))
        print(etree.tostring(elem, method="xml", encoding="utf8"))


"""
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    word_list = nltk.word_tokenize(text)
    text = [lemmatizer.lemmatize(w) for w in word_list]
    return word_list, text


def find_in_text_lemmatized(term, text,idnum):
    
    #Lemmatize term
    search_term = lemmatizer.lemmatize(term)
    #Lemmatize text
    tokens, lemmitized = lemmatize_text(text)
    #Store position of text [[start, end], ]
    position = textspan.get_original_spans(tokens, text)
    
    # Store FoundTerm object for each found instance
    found = []
#    print('term: ')
#    print(search_term)
#    print('text: ')
#    print(lemmitized)
    for i, w in enumerate(lemmitized):
        if w.lower() == search_term.lower():
            found.append(FoundTerm(term,position[i][0][0], position[i][0][1],idnum))
   
    return found

"""


def find_in_text(term, text, idnum):

    # Add spaces to term to avoid subword instances
    search_term = term
    # Store FoundTerm object for each found instance
    found = []

    for match in re.finditer(re.escape(search_term), text,
                             flags=re.IGNORECASE):
        prev_char = " "
        next_char = " "
        if match.start() != 0:
            prev_char = text[match.start() - 1]
        if match.end() < len(text) - 1:
            next_char = text[match.end()]

        if prev_char.isalnum() or next_char.isalnum():
            continue

        found.append(
            FoundTerm(match.group(), match.start(), match.end(), idnum))

    return found


def markup_term_list(element, found_terms, in_tail=False):
    elem = element
    textchunk = elem.tail if in_tail else elem.text
    num_terms = 0

    # variables to ensure things go into the right place
    parent = elem.getparent() if in_tail else None
    position = parent.index(elem) if in_tail else None
    pre = 0
    count = 1

    for item in found_terms:

        new_def_elem = etree.Element("definiendum")
        new_def_elem.text = textchunk[item.start:item.end]
        new_def_elem.tail = textchunk[item.end:]

        if item.definition_id is None:
            item.definition_id = "None"

        new_def_elem.set("id", str(item.definition_id))
        new_def_elem.set("numOccur", str(item.n_instance))

        if item.default_scope:
            new_def_elem.set("markup", "no")

        # insert the new element
        if in_tail:
            position += 1
            parent.insert(position, new_def_elem)
        else:
            elem.insert(count - 1, new_def_elem)

        # now take care of the neighbouring text
        if count == 1:
            if in_tail:
                elem.tail = textchunk[pre:item.start]
            else:
                elem.text = textchunk[pre:item.start]
        else:
            new_def_elem.getprevious().tail = textchunk[pre:item.start]

        pre = item.end
        count += 1
        num_terms += 1

    return elem, num_terms


# So as Sara, mentioned at the end of class, why don't we just replace the term with
# <definition>term<definition>. This will work, especially since we're
# only marking up strings in text. You just need to be careful about the boundaries of
# the regex, and what is captured and you'll be good to go.

# It only gets challenging when we need to interact with multiple tags and understand something about
# the subtree tags/attributes that we're working with. For this prelim markup, a replacement
# is fine.


def check_text(text_before, text_after):
    tag_starts = [(m.start(), m.end())
                  for m in re.finditer(r"<definiendum>", text_after)]
    tag_starts = [(m.start(), m.end())
                  for m in re.finditer(r"</definiendum>", text_after)]
    cropped_after = text_after[:tag_locations[0][0]]
    cropped_after += text_after[tag_locations[0][1]:tag_locations[1][0]]
    cropped_after += text_after[tag_locations[1][1]:]
    return text_before == cropped_after


# Returns a DefinedTerm object from a list of candidates with a scope containing idnum (if one exists else returns None)
#
# Parameters:
# idnum - idnum of current filepath
# candidates - list of DefinedTerm objects
# Outputs:
# candidate with a scope containing the idnum path


def getScopeMatch(idnum, candidates):

    for candidate in candidates:
        if candidate.scopeContains(idnum):
            return candidate

    return None


def markup_terms(xml_tree, definition_dictionary, idnum):

    #elems = xml_tree.xpath("//descendant-or-self::*")

    num_def_elems = 0
    old_xml = etree.tostring(xml_tree, method="xml", encoding="utf8")

    elems = xml_tree.xpath("//descendant-or-self::*")

    for (term, candidates) in definition_dictionary.items():

        candidate_match = getScopeMatch(idnum, candidates)

        if candidate_match is None:
            continue

        for elem in elems:
            if elem.text and elem.text.strip() and elem.tag != "definiendum":

                terms_in_text = find_in_text(term, elem.text, idnum)
                for found_term in terms_in_text:
                    found_term.assignDefinition(candidate_match)

                elem, num_in_text = markup_term_list(elem, terms_in_text)
                num_def_elems += num_in_text

            if elem.tail and elem.tail.strip():
                terms_in_tail = find_in_text(term, elem.tail, idnum)
                for found_term in terms_in_tail:
                    found_term.assignDefinition(candidate_match)

                elem, num_in_tail = markup_term_list(elem,
                                                     terms_in_tail,
                                                     in_tail=True)
                num_def_elems += num_in_tail

    new_xml = etree.tostring(xml_tree, method="xml", encoding="utf8")

    return new_xml


def main():

    xml_filepath = "States/kansas/chunks/ks-2021-admin-agency0001/div_kan-admin-regs-ss-1-1-1.xml"
    test_paragraph = etree.parse(xml_filepath)
    #teststring1 = "<p><level>(iii)</level> <i>Determination of whether organization is publicly supported</i> - In general ... </p>"
    #test_paragraph = etree.fromstring(teststring1)

    # What does the text look like?
    #xml_texts(test_paragraph)

    # Marking up a term, simple version, ie, no lemmatization, POS tagging, or WSD
    terms = ["Register", "civil service"]
    idnums = [0, 1]
    markup_term(test_paragraph, terms, idnums)


if __name__ == "__main__":
    main()
