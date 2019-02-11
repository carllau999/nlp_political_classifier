import string
import re
import spacy
import numpy as np
if __name__ == "__main__":
	comment = "This is ridiculous!"
	result = ""
	trimmed_result = ""
	index = 0
	f = open("../wordlists/abbrev.english", "r")
	abbrev = f.read()
	abbrev = abbrev.split()
	print("abbrev", abbrev)
	for char in comment:
		check_abbrev = False
		print(char, "extract", comment[max(0, index - 4): index + 2] )
		for a in abbrev:
			if a != "" and a in comment[max(0, index - 4): index + 2]:
				check_abbrev = True
				print("check ab", a)
		if index < len(comment) - 1 and comment[index+1] == "." and \
				check_abbrev:
			print(comment[max(0, index - 4): index + 2] , "matched abrev")
			result += char
		elif comment[min(len(comment) -1, index+1)] == "." and char.isdigit()\
				and comment[min(len(comment) -1, index+2)].isdigit():
			result += char
		elif char == "." and comment[min(len(comment) -1, index+1)].isdigit():
			result += char
		elif char in string.punctuation and ((index > 0 and comment[index
			- 1] \
				not in \
				string.punctuation) and (index < len(comment) - 1 \
						                        and comment[
					index + 1] not in string.punctuation)) and char != "'":
			print(char, "second elif")
			result += char + " "
		elif char not in string.punctuation and (index < len(comment) -1 and
				                                         comment[
				index + 1] in string.punctuation and (comment[
				index +1] != "'")):
			print(char, "third elif")
			result += char + " "
		elif char in string.punctuation and char != "'" and index < len(
				comment) -1 and comment[index + 1] not in string.punctuation:
			print(char, "fourth elif")
			result += char + " "
		else:
			print(char, "else")
			result += char
		index += 1

	space = False
	for char in result:
		if not space or char != " ":
			trimmed_result += char
		if char == " ":
			space = True
		else:
			space = False
	print("step4", trimmed_result)
	comment = trimmed_result
	index = 0
	step5 = ""
	for char in trimmed_result:
		step5 += char
		if trimmed_result[min(len(trimmed_result) - 1, index + \
				1)] == "n" and \
						trimmed_result[min(len(trimmed_result) - 1, index + \
								2)] == "'":
			step5 += " "
		elif char == "s" and trimmed_result[min(len(trimmed_result) - 1, \
				index + 1)] \
				== "'" \
				and trimmed_result[min(len(trimmed_result) -1, index + 2)] ==\
						" ":

			step5 += " "
		elif char != "n" and trimmed_result[min(len(trimmed_result) - 1, \
				index + 1)] \
				== "'" and re.match(r'[a-z]',trimmed_result[index + 2: index
				+ 3]):
			step5 += " "
		index += 1

	print("step 5", step5)
	comment = step5
	nlp = spacy.load('en', disable=['parser', 'ner'])
	doc = nlp(trimmed_result)
	tagged = ""
	for token in doc:
		tagged += token.text + "/" + token.tag_  + " "
	if len(tagged) > 0:
		tagged = tagged[0:len(tagged) - 1]
	print("tagged", tagged)
	comment = tagged
	stopwords = open("../wordlists/stopwords", "r").read().split()
	split_comment = tagged.split(" ")
	new_comment = ""
	for word in split_comment:
		if len(word.split("/", 1)) == 2:
			token = word.split("/", 1)[0]
		else:
			token = ""
		if token != "" and token.lower() not in stopwords:
			new_comment += token + "/" + word.split("/", 1)[1] + " "
	if len(new_comment) > 0:
		new_comment = new_comment[0:len(new_comment) - 1]
	print(stopwords)
	comment = new_comment
	print('remove stop words', comment)
	split_tags = comment.split(" ")
	tagged = ""
	for tg in split_tags:
		token_tag = tg.split("/", 1)
		print(token_tag)
		if len(token_tag) > 1:
			token = token_tag[0]
			tag = token_tag[1]
			token = nlp(token)[0]
			print(token.lemma_)
			if token.lemma_[0:1] == "-" and token.text[0:1] != "-":
				print("DASH", token.text, token.lemma_)
				tagged += token.text + "/" + tag + " "
			else:
				tagged += token.lemma_ + "/" + tag + " "
		else:
			print("ERROR")

	comment = tagged
	print('\nLEMMA', comment)
	split_sentences = ""
	f = open("../wordlists/abbrev.english", "r")
	abbrev = f.read().split()
	abbrev = [word.lower() for word in abbrev]
	index = 0
	# assume in tagged format
	tags = comment.split(" ")
	untag = ""
	for tg in tags:
		token_tag = tg.split("/", 1)
		if len(token_tag) > 1:
			token = token_tag[0]
			untag += token + " "
	if len(untag) > 0:
		untag = untag[0:len(untag) - 1]
	untag = untag.split(" ")
	new_line_index = []
	# hueristics from Manning and Schutze section 4.2.4
	for char in untag:
		next_char = untag[min(len(untag) - 1, index + 1)]
		if char == ".":
			phrase = untag[max(0, index - 1)] + char
			check_isabbrev = False
			for a in abbrev:
				if phrase in a:
					check_isabbrev = True
			if not check_isabbrev and index != len(untag) - 1:
				new_line_index.append(index)
		elif (char == "?" or char == "!") and next_char != " "and next_char \
				not in string.punctuation and re.match(r'[a-zA-Z]', next_char):

			new_line_index.append(index)
		index += 1
	for i in range(0, len(tags)):
		split_sentences += tags[i]
		if i in new_line_index:
			split_sentences += '\n'
		split_sentences += " "
	if len(split_sentences) > 0:
		split_sentences = split_sentences[0: len(split_sentences) - 1]
	comment = split_sentences
	print(untag)
	print(new_line_index)
	print("new line", (comment))
	FPP_1003824025 = open("../wordlists/First-person", "r").read().split()
	FPP_1003824025 = '( |^)' + '/|( |^)'.join(FPP_1003824025)
	SPP_1003824025 = open("../wordlists/Second-person", "r").read().split()
	TPP_1003824025 = open("../wordlists/Third-person", "r").read().split()
	CON_1003824025 = open("../wordlists/Conjunct", "r").read().split()
	SLANG_1003824025 = open("../wordlists/Slang", "r").read().split()
	SLANG_1003824025 = '( |^)' + '/|( |^)'.join(SLANG_1003824025)
	word_lists_1004598152 = [FPP_1003824025, SPP_1003824025, TPP_1003824025, CON_1003824025]
	# for i in range(len(word_lists_1004598152)):
	# 	word_lists_1004598152[i] = [r"(?<![^\s])" + word + "/" for word in
	# 	                            word_lists_1004598152[i]]
	# print("wordlist", word_lists_1004598152)
	comment = "I/nn am/ab amazing/afs ,/, our/a shoe/b is/cs great/non ./. " \
	          "./. ?/. a/b ?/. !/. a/nn b/nns c/nnd lmao/a wtf/i\n I/n a \n " \
	          "Awesome"
	print(FPP_1003824025)
	fpp = re.compile(r"" + FPP_1003824025, re.IGNORECASE).findall(
			comment)
	print(fpp)
	print(SLANG_1003824025)
	slang = len(re.compile(r"" + SLANG_1003824025, re.IGNORECASE).findall(
		comment))
	print(slang)
	BN_GL_1003824025 = open("../wordlists/BristolNorms+GilhoolyLogie.csv",
	                        "r").read().split("\n")
	BN_GL_1003824025 = [row.split(",") for row in BN_GL_1003824025]
	BN_GL_1003824025 = BN_GL_1003824025[1: len(BN_GL_1003824025) -1]
	BN_WORD_TO_ROW_1003824025 = {}
	for row in BN_GL_1003824025:
		BN_WORD_TO_ROW_1003824025[row[1]] = row
	# print(BN_WORD_TO_ROW_1003824025.keys())
	tokens = []
	for tg in comment.split(" "):
		tg = tg.split("/")
		if len(tg) > 1:
			token = tg[0]
			if token not in string.punctuation:
				tokens.append(token)
	avg_word_len = sum(len(word) for word in tokens) / len(tokens) if len(
		tokens) > 0 else 0
	print(tokens)
	print("avg word len", avg_word_len)
	num_sent = len(comment.split("\n"))
	avg_sent_leng = sum(len(word.split(" ")) for word in comment.split(
		"\n")) / num_sent
	print("avg sent", avg_sent_leng)
	RW_1003824025 = open("../wordlists/Ratings_Warriner_et_al.csv",
	                     "r").readlines()[1:]
	RW_1003824025 = [row.split(",") for row in RW_1003824025]
	print(RW_1003824025)
	RW_WORD_TO_ROW_1003824025 = {}
	for row in RW_1003824025:
		RW_WORD_TO_ROW_1003824025[row[1]] = row
	data = np.load('../feats/Alt_feats.dat.npy', "r")
	ALT_ID_1003824025 = open("../feats/Alt_IDs.txt", "r").read().split()
	print("nymp", ALT_ID_1003824025)
