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
package edu.emory.mathcs.nlp.mining.character.conll;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.emory.mathcs.nlp.mining.character.structure.Episode;
import edu.emory.mathcs.nlp.mining.character.structure.Scene;
import edu.emory.mathcs.nlp.mining.character.structure.Utterance;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.Tree;

/**
 * @author 	Henry(Yu-Hsin) Chen ({@code yu-hsin.chen@emory.edu})
 * @version	1.0
 * @since 	Jan 18, 2017
 */
public class JSONtoConllFormatPrinter {
	private static Set<String> ignore_referents = new HashSet<>(Arrays.asList("General", "Collective", "Other"));
	
	public void printScene(PrintWriter printer, Scene scene, String docuement_name, boolean gold) {
		printScene(printer, scene, docuement_name, 0, gold);
	}
	
	public void printScene(PrintWriter printer, Scene scene, String docuement_name, int part_id, boolean gold) {
		printer.printf("#begin document (%s); part %03d\n", docuement_name, part_id);
		
		int cid = 0; Sentence sentence; 
		String[][] coref_labels = gold? toConllCorefLabels(scene) : null;
		for(Utterance utternace : scene.getUtterances()) {
			String speaker = utternace.speaker;
			
			for(List<String> statement : utternace.tokenized_statement_sentences){
				sentence = new Sentence(statement);
				
				List<String> ws = sentence.words(), p = sentence.posTags(), n = sentence.nerTags(), l = sentence.lemmas();
				String[] tokens = ws.stream().toArray(String[]::new), pos = p.stream().toArray(String[]::new), lemma = l.stream().toArray(String[]::new),
						ner = toConllNERLabels(n), ct = toConllCTLabels(ws, p, sentence.parse()), coref = gold? coref_labels[cid++] : null;
						
				for(int tid = 0; tid < tokens.length; tid++)
					if(gold) printer.printf("%s %3d %3d %15s %5s %25s %15s     -     - %15s %10s %10s\n", docuement_name, part_id, 
								tid, tokens[tid], pos[tid], ct[tid], lemma[tid], speaker, ner[tid], coref[tid]);
					else	 printer.printf("%s %3d %3d %15s %5s %25s %15s     -     - %15s %10s %10s\n", docuement_name, part_id, 
								tid, tokens[tid], pos[tid], ct[tid], lemma[tid], speaker, ner[tid], "-");
				
				printer.println();
			}
		}
		printer.printf("#end document\n");
		printer.flush();
	}
	
	public void printEpisode(PrintWriter printer, Episode episode, String docuement_name, boolean gold, boolean scene_delim) {
		int sid = 0, cid = 0; String[][] coref_labels = gold? toConllCorefLabels(episode) : null;
		int mention_count = 0; Sentence sentence; StringBuilder sb = new StringBuilder();
		
		if(!scene_delim) 
			sb.append(String.format("#begin document (%s); part %03d\n", docuement_name, sid));
		
		for(Scene scene : episode.getScenes()) {
			if(scene.size() <= 0) continue;
			
			if(scene_delim) {
				mention_count = 0; sb.setLength(0);
				sb.append(String.format("#begin document (%s); part %03d\n", docuement_name, sid));
			}
			
			for(Utterance utternace : scene.getUtterances()) {
				String speaker = utternace.speaker;
				
				for(List<String> statement : utternace.tokenized_statement_sentences){
					sentence = new Sentence(statement);
					
					List<String> ws = sentence.words(), p = sentence.posTags(), n = sentence.nerTags(), l = sentence.lemmas();
					String[] tokens = ws.stream().toArray(String[]::new), pos = p.stream().toArray(String[]::new), lemma = l.stream().toArray(String[]::new),
							ner = toConllNERLabels(n), ct = toConllCTLabels(ws, p, sentence.parse()), coref = gold? coref_labels[cid++] : null;
					
					for(int tid = 0; tid < tokens.length; tid++) {
						if(gold) sb.append(String.format("%s %3d %3d %15s %5s %25s %15s     -     - %15s %10s %10s\n", docuement_name, sid, 
								tid, tokens[tid], pos[tid], ct[tid], lemma[tid], speaker, ner[tid], coref[tid]));
						else	 sb.append(String.format("%s %3d %3d %15s %5s %25s %15s     -     - %15s %10s %10s\n", docuement_name, sid, 
								tid, tokens[tid], pos[tid], ct[tid], lemma[tid], speaker, ner[tid], "-"));
						
						if(!coref[tid].equals("-")) mention_count++;
					}
					sb.append('\n');
				}
			}
			
			if(scene_delim && mention_count > 0) {
				printer.printf("%s#end document\n", sb.toString()); sid++; 
			}
		}
		if(!scene_delim && mention_count > 0)  
			printer.printf("%s#end document\n", sb.toString());
		
		printer.flush();
	}
	
	private String[] toConllCTLabels(List<String> words, List<String> pos_tags, Tree tree) {
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
	
	private String[] toConllNERLabels(List<String> ner_tags) {
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
			else new_tags.add("*");
		}
		
		if(new_tags.size() != ner_tags.size())
			throw new RuntimeException("Mismatch in word converted NER label and counts.");
		
		return new_tags.stream().toArray(String[]::new);
	}

	private String[][] toConllCorefLabels(Scene scene) {
		List<String[]> statements = new ArrayList<>();
		
		for(Utterance utterance : scene){
			for(List<String> statement : utterance.statement_sentence_annotations)
				statements.add(statement.stream().toArray(String[]::new));
		}
		return toConllCorefLabels(statements.stream().toArray(String[][]::new));
	}
	
	private String[][] toConllCorefLabels(Episode episode) {
		List<String[]> statements = new ArrayList<>();
		
		for(Scene scene : episode.getScenes()){
			for(Utterance utterance : scene){
				for(List<String> statement : utterance.statement_sentence_annotations)
					statements.add(statement.stream().toArray(String[]::new));
			}
		}
		return toConllCorefLabels(statements.stream().toArray(String[][]::new));
	}
	
	private String[][] toConllCorefLabels(String[][] annotations) {
		int i, j, idx; Map<String, Integer> m_clusterId = new HashMap<>();
		String[][] labels = new String[annotations.length][];
		
		for(i = idx = 0; i < annotations.length; i++){
			labels[i] = new String[annotations[i].length];
			Arrays.fill(labels[i], "-");
			
			for(j = 0; j < annotations[i].length; j++) {
				String annotation = annotations[i][j];
				
				char bilou = annotation.charAt(0);
				if((bilou == 'B' || bilou == 'U') && annotation.length() > 2) {
					String referent = annotation.substring(2);
					
					if(!ignore_referents.contains(referent) && !m_clusterId.containsKey(referent)) 
						m_clusterId.put(referent, idx++);
				}
			}			
		}
		
		for(i = 0; i < annotations.length; i++){
			for(j = 0; j < annotations[i].length; j++) {
				String annotation = annotations[i][j];
				
				char bilou = annotation.charAt(0);
				if(annotation.length() > 2) {
					int id = -1; String referent = annotation.substring(2);
					
					
					if(m_clusterId.containsKey(referent)) 	id = m_clusterId.get(referent);
					if (id >= 0){
						switch (bilou) {
						case 'B':	labels[i][j] = String.format( "(%d", id);	break;
						case 'L':	labels[i][j] = String.format( "%d)", id);	break;	
						case 'U':	labels[i][j] = String.format("(%d)", id);	break;
						}
					}
					else if(referent.equals("Other")) {
						id = ++idx;
						
						if(bilou == 'U') labels[i][j] = String.format("(%d)", id);
						else if(bilou == 'B'){
							labels[i][j] = String.format( "(%d", id);
							
							for(j++; j < annotations[i].length; j++) {
								annotation = annotations[i][j]; bilou = annotation.charAt(0);
								if(bilou == 'L') { labels[i][j] = String.format( "%d)", id); break; }
							}
						}
					}
				}
			}
		}
		return labels;
	}
}
