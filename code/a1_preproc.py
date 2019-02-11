import sys
import argparse
import os
import json
import re
import html
import string
import spacy

indir = '/u/cs401/A1/data/'
nlp = spacy.load('en', disable=['parser', 'ner'])
abbrev_10003824025 = open("/u/cs401/Wordlists/abbrev.english", "r").read().split()
stopwords_1003824025 = open("/u/cs401/Wordlists/StopWords", "r").read().split()

def preproc1(comment, steps=range(1, 11)):
	''' This function pre-processes a single comment

	Parameters:
		comment : string, the body of a comment
		steps   : list of ints, each entry in this list corresponds to a preprocessing step

	Returns:
		modComm : string, the modified comment
	'''

	if 1 in steps:
		comment = comment.replace("\n", " ")
	if 2 in steps:
		comment = html.unescape(comment)
	if 3 in steps:
		comment = re.sub(r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?['
		                 'A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/['
		                 '\+~%\/.\w_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:['
		                 '\w]*))?)', "", comment)
	if 4 in steps:
		result = ""
		trimmed_result = ""
		index = 0
		for char in comment:
			check_abbrev = False
			# check if char is part of an abbreviation
			for a in abbrev_10003824025:
				if a != "" and a in comment[max(0, index - 4): index + 2]:
					check_abbrev = True
			# if part of abbrev and behind a . then do not add whitespace
			if index < len(comment) - 1 and comment[index + 1] == "." and check_abbrev:
				result += char
			# if part of a floating number and behind a . then do not add whitespace
			elif comment[min(len(comment) - 1, index + 1)] == "." and char.isdigit() and comment[min(len(comment) - 1,
			                                                                                    index + 2)].isdigit():
				result += char
			# if part of floating number and is a . then do not add white space
			elif char == "." and comment[min(len(comment) - 1, index + 1)].isdigit():
				result += char
			# if a punctuation and not part of a group of punctuation then add whitespace
			elif char in string.punctuation and ((index > 0 and comment[index - 1] not in string.punctuation) and
				(index < len(comment) - 1 and comment[index + 1] not in string.punctuation)) and char != "'":
				result += char + " "
			# if not a punctuation and behind one then add a whitespace
			elif char not in string.punctuation and (index < len(comment) - 1 and comment[index + 1]
				in string.punctuation and (comment[index + 1] != "'")):
				result += char + " "
			# if a punctuation a not behind one then add a white space
			elif char in string.punctuation and char != "'" and index < len(comment) - 1 and comment[index + 1] \
					not in string.punctuation:
				result += char + " "
			else:
				result += char
			index += 1
		space = False
		# remove extra spacing
		for char in result:
			if not space or char != " ":
				trimmed_result += char
			if char == " ":
				space = True
			else:
				space = False
		comment = trimmed_result
	if 5 in steps:
		index = 0
		result = ""
		for char in comment:
			result += char
			# if behind a n' then add white space
			if comment[min(len(comment) - 1, index + 1)] == "n" and comment[min(len(comment) - 1, index + 2)] == "'":
				result += " "
			# if char and following two characterse in front match s' with space then add whitespace
			elif char == "s" and comment[min(len(comment) - 1, index + 1)] == "'" and comment[min(len(comment) - 1,
																								index + 2)] == " ":

				result += " "
			# if behind a ' and a letter then a whitespace
			elif char != "n" and comment[min(len(comment) - 1, index + 1)] == "'" and re.match(r'[a-z]',
																					comment[index + 2: index+ 3]):
				result += " "
			index += 1
		comment = result
	if 6 in steps:
		doc = nlp(comment)
		tagged = ""
		# tag tokens with spacy tag_
		for token in doc:
			tagged += token.text + "/" + token.tag_ + " "
		if len(tagged) > 0:
			tagged = tagged[0:len(tagged) - 1]
		comment = tagged
	if 7 in steps:
		# assume in tagged format
		split_comment = comment.split(" ")
		new_comment = ""
		for word in split_comment:
			if len(word.split("/", 1)) == 2:
				token = word.split("/", 1)[0]
			else:
				token = ""
			if token != "" and token.lower() not in stopwords_1003824025:  # remove stopwords
				new_comment += token + "/" + word.split("/", 1)[1] + " "
		if len(new_comment) > 0:
			new_comment = new_comment[0:len(new_comment) - 1]  # remove extra space
		comment = new_comment
	if 8 in steps:
		# assume in tagged format
		split_tags = comment.split(" ")
		tagged = ""
		# convert tokens to lemma
		for tg in split_tags:
			token_tag = tg.split("/", 1)
			if len(token_tag) > 1:
				token = token_tag[0]
				tag = token_tag[1]
				token = nlp(token)[0]
				if token.lemma_[0:1] == "-" and token.text[0:1] != "-":
					tagged += token.text + "/" + tag + " "
				else:
					tagged += token.lemma_ + "/" + tag + " "
		comment = tagged

	if 9 in steps:
		split_sentences = ""
		abbrev = [word.lower() for word in abbrev_10003824025]
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
						check_isabbrev = True  # check if phrase is an abbreviation
				# if char is . and not within multiple punctuation and not an abbreviation then the sentence ends
				if not check_isabbrev and index != len(untag) - 1 and next_char not in string.punctuation:
					new_line_index.append(index)
			# if char is ? or ! and is behind letters then the sentence has ended
			elif (char == "?" or char == "!") and next_char != " " and next_char not in string.punctuation and\
																		re.match(r'[a-zA-Z]', next_char):

				new_line_index.append(index)
			index += 1
		# add \n to the selected indicies
		for i in range(0, len(tags)):
			split_sentences += tags[i]
			if i in new_line_index:
				split_sentences += '\n'
			split_sentences += " "
		if len(split_sentences) > 0:
			split_sentences = split_sentences[0: len(split_sentences) - 1]
		comment = split_sentences

	if 10 in steps:
		# convert all to lowercase
		comment = comment.lower()

	modComm = comment
	return modComm


def main(args):
	allOutput = []
	for subdir, dirs, files in os.walk(indir):
		for file in files:
			fullFile = os.path.join(subdir, file)
			print("Processing " + fullFile)

			data = json.load(open(fullFile))
			p1 = args.ID[0] % len(data)
			# circular list slice to ensure arg.max is extracted
			if p1 + int(args.max) >= len(data):
				p2 = len(data) - p1
				missing = int(args.max) - p2
				extracted_data = data[p1:] + data[:missing]
			else:
				p2 = p1 + int(args.max)
				extracted_data = data[p1: p2]
			for d in extracted_data:
				data = json.loads(d)
				body = preproc1(data["body"])
				id_ = data["id"]
				new_data = {"body": body, "id": id_, "cat": file}
				allOutput.append(new_data)

	fout = open(args.output, 'w')
	fout.write(json.dumps(allOutput))
	fout.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Process each .')
	parser.add_argument('ID', metavar='N', type=int, nargs=1,
	                    help='your student ID')
	parser.add_argument("-o", "--output",
	                    help="Directs the output to a filename of your choice",
	                    required=True)
	parser.add_argument("--max",
	                    help="The maximum number of comments to read from each file",
	                    default=10000)
	args = parser.parse_args()

	if (int(args.max) > 200272):
		print(
			"Error: If you want to read more than 200,272 comments per file, you have to read them all.")
		sys.exit(1)

	main(args)
