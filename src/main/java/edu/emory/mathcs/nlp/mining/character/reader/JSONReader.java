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
package edu.emory.mathcs.nlp.mining.character.reader;

import java.io.BufferedReader;
import java.io.InputStream;

import com.google.gson.Gson;

import edu.emory.mathcs.nlp.common.util.IOUtils;
import edu.emory.mathcs.nlp.mining.character.structure.Season;

/**
 * @author 	Henry(Yu-Hsin) Chen ({@code yu-hsin.chen@emory.edu})
 * @version	1.0
 * @since 	Jan 18, 2017
 */
public class JSONReader {
	private Gson gson = new Gson();
	
	public Season readSeason(InputStream in) {
		BufferedReader reader = IOUtils.createBufferedReader(in);
		return gson.fromJson(reader, Season.class);
	}
	
	
}
