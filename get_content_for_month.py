import os
import sys
import pandas as pd
import requests
import re
import time
from tqdm import tqdm

import mwparserfromhell as mwp

CONTENT_URL = "https://en.wikipedia.org/w/api.php?action=parse&oldid={0}&prop=wikitext&formatversion=2&format=json"


def _parse_and_clean_wikicode(raw_content):
	"""Strips formatting and unwanted sections from raw page content."""
	wikicode = mwp.parse(raw_content)

	# Filters for references, tables, and file/image links.
	re_rm_wikilink = re.compile(
		"^(?:File|Image|Media):", flags=re.IGNORECASE | re.UNICODE
	)

	def rm_wikilink(obj):
		return bool(re_rm_wikilink.match(str(obj.title)))  # pytype: disable=wrong-arg-types

	def rm_tag(obj):
		return str(obj.tag) in {"ref", "table"}

	def rm_template(obj):
		return obj.name.lower() in {
			"reflist",
			"notelist",
			"notelist-ua",
			"notelist-lr",
			"notelist-ur",
			"notelist-lg",
		}

	def try_remove_obj(obj, section):
		try:
			section.remove(obj)
		except ValueError:
			# For unknown reasons, objects are sometimes not found.
			pass

	section_text = []
	# Filter individual sections to clean.
	for section in wikicode.get_sections(
			flat=True, include_lead=True, include_headings=True
	):
		for obj in section.ifilter_wikilinks(matches=rm_wikilink, recursive=True):
			try_remove_obj(obj, section)
		for obj in section.ifilter_templates(matches=rm_template, recursive=True):
			try_remove_obj(obj, section)
		for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
			try_remove_obj(obj, section)

		section_text.append(section.strip_code().strip())
	return "\n\n".join(section_text)

def get_valid_filename(s):
	s = str(s).strip().replace(' ', '_')
	return re.sub(r'(?u)[^-\w.]', '', s)

def get_content_for_revision_id(revision_id, last_try: bool = False):
	try:
		retries = 3
		while retries > 0:
			content = requests.get(url=CONTENT_URL.format(revision_id)).json()
			if "parse" not in content:
				print(f"Failed to find for {revision_id}, retrying")
				retries -= 1
				time.sleep(1)
			else:
				return content["parse"]["wikitext"]
			
	except Exception as e:
		time.sleep(1)
		if not last_try:
			print(f"Retrying")
			return get_content_for_revision_id(revision_id, last_try=True)
		else:
			print(f"Failed to find for {revision_id}, giving up")
			return None    

def get_content_for_article(row, save_path):
	for month_no, month in enumerate(row.index):
		if month_no <= 1:
			continue

		if row[month] != -1:
			content = get_content_for_revision_id(row[month])
			clean_content = _parse_and_clean_wikicode(content).replace("===", " ").replace("==", " ")

			save_folder = "-".join(month.split("-")[:2])
			if not os.path.exists(f"{save_path}/{save_folder}"):
				os.makedirs(f"{save_path}/{save_folder}")

			with open(f"{save_path}/{save_folder}/{get_valid_filename(row['title'])}-{int(row[month])}.txt", "w", encoding="utf-8") as f:
				f.write(row["title"] + "\n\n" + clean_content)

if __name__ == "__main__":
	assert len(sys.argv) > 2, "Provide a file with revision ids and a save path location"
	content_file_path = sys.argv[1]
	save_path = sys.argv[2]

	tqdm.pandas()

	df = pd.read_csv(content_file_path)
	df.progress_apply(lambda x: get_content_for_article(x, save_path), axis=1)
