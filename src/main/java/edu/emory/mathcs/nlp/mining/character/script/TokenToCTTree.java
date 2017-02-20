/**
 * Copyright 2016, Emory University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.emory.mathcs.nlp.mining.character.script;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.Tree;

/**
 * @author 	Henry(Yu-Hsin) Chen ({@code yu-hsin.chen@emory.edu})
 * @version	1.0
 * @since 	Jan 17, 2017
 */
public class TokenToCTTree {

	public static void main(String[] args) {
		Sentence sentence = new Sentence(Arrays.asList("Lucy", "Chu", "is", "in", "the", "sky", "with", "diamonds", "on", "July", "3", "."));
		
		List<String> pos_tags = sentence.posTags();
		List<String> ner_tags = sentence.nerTags();
		Tree ct  = sentence.parse();
		
		System.out.println(sentence.words()); System.out.println(pos_tags); 
		System.out.println(toConllNERLabels(ner_tags));
		System.out.println(toConllCTLabels(sentence.words(), pos_tags, ct));
	}
	
	public static String[] toConllCTLabels(List<String> words, List<String> pos_tags, Tree tree) {
		String penn_label = tree.pennString().replaceAll("\n", " ").replaceAll("[ ]{2,}", " ");
		StringBuilder sb = new StringBuilder();
		
		if(words.size() != pos_tags.size())
			throw new IllegalArgumentException("Mismathc in token and post tag counts.");
		
		for(int idx = 0; idx < words.size(); idx++) {
			String chunck = String.format("(%s %s)", pos_tags.get(idx), words.get(idx));
			int start = penn_label.indexOf(chunck), end = start + chunck.length();
			
			if(start > 0){
				sb.append(penn_label.substring(0, start));
				sb.append('*'); penn_label = penn_label.substring(end);
			}
		}
		sb.append(penn_label);
		String line = sb.toString().replace("ROOT", "TOP");
		
		List<String> labels = new ArrayList<>();
		for(int idx = 0; idx < words.size(); idx++) {
			int offset = line.indexOf('*')+1;
			
			while(++offset < line.length() && line.charAt(offset) != '*' && line.charAt(offset) != '(');
			
			labels.add(line.substring(0, offset).replaceAll(" ", ""));
			line = line.substring(offset);
		}
		
		if(labels.size() != words.size())
			throw new RuntimeException("Mismatch in word and converted parsed tree label counts.");
		
		return labels.stream().toArray(String[]::new);
	}
	
	public static String[] toConllNERLabels(List<String> ner_tags) {
		List<String> new_tags = new ArrayList<>();
		 
		for(int idx = 0, end; idx < ner_tags.size(); idx++) {
			String current = ner_tags.get(idx);
			
			if(!current.equals("O")) {
				for(end = idx+1; end < ner_tags.size() && ner_tags.get(end).equals(current); end++);
				
				if(end == idx+1) new_tags.add(String.format("(%s)", current));
				else if(end > idx+1) {
					new_tags.add(String.format("(%s*", current));
					for(int c = idx; c < end-2; c++) new_tags.add("*");
					new_tags.add(String.format("*)", current));
				}
				idx = end-1;
			}
			else new_tags.add(current);
		}
		
		if(new_tags.size() != ner_tags.size())
			throw new RuntimeException("Mismatch in word converted NER label and counts.");
		
		return new_tags.stream().toArray(String[]::new);
	}
}
