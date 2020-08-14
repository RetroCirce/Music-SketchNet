import pretty_midi as pyd
import numpy as np
import os
import random
import music21 as m21
from loader.chordloader import Chord_Loader
import copy

class MIDI_Loader:
    def __init__(self, datasetName, minStep = 0.03125):
        self.datasetName = datasetName
        self.minStep = minStep
        self.min_step = minStep
    def load(self, directory):
        path = os.listdir(directory)
        print("Dataset Name: " + self.datasetName)
        print("start to load mid from %s" % directory)
        # Nottingham dataset is processed by y
        self.midi_files = [] 
        self.directory = directory
        total = 0
        if self.datasetName == "Nottingham":
            for midi_file in path:
                self.midi_files.append({"name": (midi_file.split("."))[0], "raw": pyd.PrettyMIDI(directory + midi_file)})
                if total % 1000 == 0:
                    print("processed:%d" % (total + 1))
                total = total + 1
        if self.datasetName == "Irish":
            for midi_file in path:
                if not midi_file.endswith(".mid"):
                    continue
                temp = pyd.PrettyMIDI(directory + midi_file)
                # useless midi file
                if len(temp.instruments) == 0 or len(temp.instruments[0].notes) == 0:
                    continue
                tsc = temp.time_signature_changes
                if len(tsc) == 1 and tsc[0].numerator == 4 and tsc[0].denominator == 4:
                    self.midi_files.append({"name": (midi_file.split("."))[0], "raw": temp})
                    if total % 1000 == 0:
                        print("processed:%d" % (total + 1))
                    total = total + 1
        print("loading %s success! %d files in total" %(directory, len(self.midi_files)))
        return None
    def getChordSeq(self, recogLevel = "Mm"):
        print("start to get chord sequences")
        self.recogLevel = recogLevel
        if self.datasetName == "Nottingham":
            # 25 one hot vectors
            # 0-11 for major
            # 12-23 for minor 
            # 24 for NC
            for i in range(len(self.midi_files)):
                midi_data = self.midi_files[i]["raw"]
                cl = Chord_Loader(recogLevel = self.recogLevel)
                chord_set = []
                chord_time = [0.0, 0.0]
                last_time = 0.0 # Define the chord recognition system
                chord_file = []
                if len(midi_data.instruments) == 1:
                    self.midi_files[i]["chords"] = {}
                    self.midi_files[i]["chord_seq"] = []
                    continue
                self.midi_files[i]["chords"] = []
                for note in midi_data.instruments[1].notes:
                    if len(chord_set) == 0:
                        chord_set.append(note.pitch)
                        chord_time[0] = note.start
                        chord_time[1] = note.end
                    else:
                        if note.start == chord_time[0] and note.end == chord_time[1]:
                            chord_set.append(note.pitch)
                        else:
                            if last_time < chord_time[0]:
                                self.midi_files[i]["chords"].append({"start":last_time ,"end": chord_time[0], "chord" : "NC"})
                            self.midi_files[i]["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": cl.note2name(chord_set)})
                            last_time = chord_time[1]
                            chord_set = []
                            chord_set.append(note.pitch)
                            chord_time[0] = note.start
                            chord_time[1] = note.end 
                if chord_set:
                    if last_time < chord_time[0]:
                        self.midi_files[i]["chords"].append({"start":last_time ,"end": chord_time[0], "chord" : "NC"})
                    self.midi_files[i]["chords"].append({"start":chord_time[0],"end":chord_time[1],"chord": cl.note2name(chord_set)})
                    last_time = chord_time[1]
                for c in self.midi_files[i]["chords"]:
                    c_index = cl.name2index(c["chord"])
                    steps = int((c["end"] - c["start"]) / self.minStep)
                    for j in range(steps):
                        chord_file.append(c_index)
                self.midi_files[i]["chord_seq"] = chord_file
            print("calc chords success! %d files in total" % len(self.midi_files))
            return self.midi_files
        if self.datasetName == "Irish":
            print("Error:Irish Folk Song dataset has no chord")
            return None
        print("Error: No dataset called " +  self.datasetName)
        return None
    def getNoteSeq(self):
        print("start to get notes")
        if self.datasetName == "Nottingham":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            for i in range(len(self.midi_files)):
                midi_data = self.midi_files[i]["raw"]
                pitch_file = []
                cst = 0.0
                cet = midi_data.instruments[0].notes[0].start
                cpitch = rest_pitch
                flag = False
                for note in midi_data.instruments[0].notes:
                    flag = True
                    if note.start > cst:
                        flag = False
                        if note.start > cet:
                            steps = int((cet - cst) / self.minStep)                              
                            pitch_file.append(cpitch)
                            add_pitch = rest_pitch if (cpitch == rest_pitch) else hold_pitch
                            for j in range(steps - 1):
                                pitch_file.append(add_pitch)
                            steps = int((note.start - cet) / self.minStep)
                            pitch_file.append(rest_pitch)
                            for j in range(steps - 1):
                                pitch_file.append(rest_pitch)
                        else:
                            steps = int((note.start - cst) / self.minStep)
                            pitch_file.append(cpitch)
                            add_pitch = rest_pitch if (cpitch == rest_pitch) else hold_pitch
                            for j in range(steps - 1):
                                pitch_file.append(add_pitch)
                        cst = note.start
                        cet = note.end
                        cpitch = note.pitch
                if flag == False:
                    steps = int((cet - cst) / self.minStep)
                    pitch_file.append(cpitch)
                    add_pitch = rest_pitch if (cpitch == rest_pitch) else hold_pitch
                    for j in range(steps - 1):
                        pitch_file.append(add_pitch)
                self.midi_files[i]["notes"] = pitch_file[:]
            print("calc notes success! %d files in total" % len(self.midi_files))
            return self.midi_files
        if self.datasetName == "Irish":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            c_bias = 1.0 / 960
            for i in range(len(self.midi_files)):
                midi_data = self.midi_files[i]["raw"]
                pitch_file = []
                cst = midi_data.instruments[0].notes[0].start - c_bias
                # first note starts > 0
                if cst > 0:
                    steps = int(round(cst / self.min_step))
                    pitch_file.extend([rest_pitch] * steps)
                cst = 0.0
                cet = midi_data.instruments[0].notes[0].start - c_bias
                cpitch = rest_pitch
                flag = False
                for note in midi_data.instruments[0].notes:
                    # print(abs(note.start - c_bias - cet))
                    if note.start - cet - c_bias >= -0.0001:
                        den_step = int(round((note.start - cet - c_bias) / self.min_step))
                        pitch_file.extend([rest_pitch] * den_step)
                        cst = note.start - c_bias
                        cet = note.end
                        add_pitch = note.pitch
                        steps = int(round((cet - cst) / self.min_step))
                        if steps > 0:
                            pitch_file.extend([add_pitch])
                            pitch_file.extend([hold_pitch] * (steps - 1))
                    elif note.start - c_bias <= cst:
                        if not flag: 
                            print("find ahead notes in Irish Folk Song dataset in %s" %(self.midi_files[i]["name"]))
                        flag = True
                        continue
                self.midi_files[i]["notes"] = pitch_file[:]
                if i % 1000 == 0:
                    print("get notes in %d files" % (i + 1))
            print("calc notes success! %d files in total" % len(self.midi_files))
            return self.midi_files
        print("Error: No dataset called " +  self.datasetName)
        return False
    def getChordFunctions(self):
        return True
    def dataAugment(self,bottom = 40, top = 85):
        print("start to augment data")
        #print("Be sure you get the chord functions before!")
        augment_data = []
        if self.datasetName == "Nottingham":
            cl = Chord_Loader(recogLevel = self.recogLevel)
            for i in range(-5,7,1):
                for x in self.midi_files:
                    midi_file = copy.deepcopy(x)
                    is_add = True
                    for j in range(len(midi_file["notes"])):
                        if midi_file["notes"][j] <= 127:
                            midi_file["notes"][j] += i
                            if midi_file["notes"][j] > top or midi_file["notes"][j] < bottom:
                                is_add = False
                                break
                    for j in range(len(midi_file["chord_seq"])):
                        midi_file["chord_seq"][j] = cl.chord_alu(x = midi_file["chord_seq"][j],scalar = i)
                    if is_add:
                        midi_file["name"] += "-switch(" + str(i) + ")" 
                        augment_data.append(midi_file)
                print("finish augment %d data" % i)
            self.midi_files = augment_data
            # random.shuffle(self.midi_files)
            print("data augment success! %d files in total" % len(self.midi_files))
            return self.midi_files
        print("Error: No dataset called " +  self.datasetName)
        return False
    def getData(self):
        return self.midi_files
    def processed_all(self):
        if self.datasetName == "Irish":
            print("start process Irish Folk Song dataset")
            self.getNoteSeq()
            print("processing succeed")
            return self.midi_files
    def writeFile(self, output = ""):
        print("begin write file from %s" % self.directory)
        for midi_file in self.midi_files:
            output_file = []
            if midi_file.__contains__("name"):
                output_file.append("Name: " + midi_file["name"] + "\n")
            if midi_file.__contains__("chord_seq"):
                output_file.append("Chord Sequence:\n")
                for c in midi_file["chord_seq"]:
                    output_file.append(str(c) + " ")
                output_file.append("\n")
            if midi_file.__contains__("notes"):
                output_file.append("Notes:\n")
                for c in midi_file["notes"]:
                    output_file.append(str(c) + " ")
                output_file.append("\n")
            with open(output + midi_file["name"] + ".txt","w") as f:
                f.writelines(output_file)
        print("finish output! %d files in total" % len(self.midi_files))
        return True

class MIDI_Render:
    def __init__(self, datasetName, minStep = 0.03125):
        self.datasetName = datasetName
        self.minStep = minStep
    def data2midi(self, data, recogLevel = "Mm", output = "test.mid"):
        gen_midi = pyd.PrettyMIDI()
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        chords = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        if self.datasetName == "Nottingham":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            cl = Chord_Loader(recogLevel = recogLevel)
            time_shift = 0.0
            local_duration = 0
            prev = "NC"
            for chord in data["chords"]:
                if chord == "":
                    continue
                chord = cl.index2name(x = int(chord))
                if chord == prev:
                    local_duration += 1
                else:
                    if prev == "NC":
                        prev = chord
                        time_shift += local_duration * self.minStep
                        local_duration = 1
                    else:
                        i_notes = cl.name2note(name = prev, stage = 4)
                        for i_note in i_notes:
                            i_note = pyd.Note(velocity = 100, pitch = i_note, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                            chords.notes.append(i_note)
                        prev = chord
                        time_shift += local_duration * self.minStep
                        local_duration = 1
            if prev != "NC":
                i_notes = cl.name2note(name = prev, stage = 4)
                for i_note in i_notes:
                    i_note = pyd.Note(velocity = 100, pitch = i_note, 
                    start = time_shift, end = time_shift + local_duration * self.minStep)
                    chords.notes.append(i_note)
            gen_midi.instruments.append(chords)

            time_shift = 0.0
            local_duration = 0
            prev = rest_pitch
            for note in data["notes"]:
                note = int(note)
                if note < 0 or note > 129:
                    continue
                if note == hold_pitch:
                    local_duration += 1
                elif note == rest_pitch:
                    time_shift += self.minStep
                else:
                    if prev == rest_pitch:
                        prev = note
                        local_duration = 1
                    else:
                        i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                        melodies.notes.append(i_note)
                        prev = note
                        time_shift += local_duration * self.minStep
                        local_duration = 1
            if prev != rest_pitch:
                i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                melodies.notes.append(i_note)
            gen_midi.instruments.append(melodies)
            gen_midi.write(output)
            print("finish render midi on " + output)
        if self.datasetName == "Irish":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            local_duration = 0
            time_shift = 0.0
            local_duration = 0
            prev = rest_pitch
            for note in data["notes"]:
                note = int(note)
                if note < 0 or note > 129:
                    continue
                if note == hold_pitch:
                    local_duration += 1
                elif note == rest_pitch:
                    time_shift += self.minStep
                else:
                    if prev == rest_pitch:
                        prev = note
                        local_duration = 1
                    else:
                        i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                        melodies.notes.append(i_note)
                        prev = note
                        time_shift += local_duration * self.minStep
                        local_duration = 1
            if prev != rest_pitch:
                i_note = pyd.Note(velocity = 100, pitch = prev, 
                            start = time_shift, end = time_shift + local_duration * self.minStep)
                melodies.notes.append(i_note)
            gen_midi.instruments.append(melodies)
            gen_midi.write(output)
            print("finish render midi on " + output)
            
    def text2midi(self, text_ad, recogLevel = "Mm",output = "test.mid"):
        gen_midi = pyd.PrettyMIDI()
        melodies = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        chords = pyd.Instrument(program = pyd.instrument_name_to_program('Acoustic Grand Piano'))
        if self.datasetName == "Nottingham":
            # 130 one hot vectors 
            # 0-127 for pitch
            # 128 for hold 129 for rest
            rest_pitch = 129
            hold_pitch = 128
            with open(text_ad,"r") as f:
                lines = f.readlines()
                read_flag = "none"
                for line in lines:
                    line = line.strip()
                    # if line == "Chord:":
                    #     continue
                    if line == "Chord Sequence:":
                        read_flag = "chord_seq"
                        continue
                    if line == "Notes:":
                        read_flag = "notes"
                        continue
                    if read_flag == "chord_seq":
                        cl = Chord_Loader(recogLevel = recogLevel)
                        elements = line.split(" ")
                        time_shift = 0.0
                        local_duration = 0
                        prev = "NC"
                        for chord in elements:
                            if chord == "":
                                continue
                            chord = cl.index2name(x = int(chord))
                            if chord == prev:
                                local_duration += 1
                            else:
                                if prev == "NC":
                                    prev = chord
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                                else:
                                    i_notes = cl.name2note(name = prev, stage = 4)
                                    for i_note in i_notes:
                                        i_note = pyd.Note(velocity = 100, pitch = i_note, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                                        chords.notes.append(i_note)
                                    prev = chord
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                        if prev != "NC":
                            i_notes = cl.name2note(name = prev, stage = 4)
                            for i_note in i_notes:
                                i_note = pyd.Note(velocity = 100, pitch = i_note, 
                                start = time_shift, end = time_shift + local_duration * self.minStep)
                                chords.notes.append(i_note)
                        gen_midi.instruments.append(chords)
                        continue
                    if read_flag == "notes":
                        elements = line.split(" ")
                        time_shift = 0.0
                        local_duration = 0
                        prev = rest_pitch
                        for note in elements:
                            note = int(note)
                            if note < 0 or note > 129:
                                continue
                            if note == hold_pitch:
                                local_duration += 1
                            elif note == rest_pitch:
                                time_shift += self.minStep
                            else:
                                if prev == rest_pitch:
                                    prev = note
                                    local_duration = 1
                                else:
                                    i_note = pyd.Note(velocity = 100, pitch = prev, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                                    melodies.notes.append(i_note)
                                    prev = note
                                    time_shift += local_duration * self.minStep
                                    local_duration = 1
                        if prev != rest_pitch:
                            i_note = pyd.Note(velocity = 100, pitch = prev, 
                                        start = time_shift, end = time_shift + local_duration * self.minStep)
                            melodies.notes.append(i_note)
                        gen_midi.instruments.append(melodies)
                        continue
                gen_midi.write(output)
                print("finish render midi on " + output)
                                

class DataLoader:
    def __init__(self, train = None, validate = None, test = None):
        self.train = train
        self.validate = validate
        self.test = test
        self.hold_pitch = 128
        self.batch_idx = 0
        self.batch_size = 64
        print("dataset length \tTrain: %d\tVali: %d\tTest: %d" %(len(train),
        len(validate),len(test)))
        print("###### data example")
        print(train[0])
        print(validate[0])
        print(test[0])
        print("################")
    def process_split(self, split_size = 24):
        self.train_set = []
        self.validate_set = []
        self.test_set = []
        print("begin processing train:")
        i = 0
        for d in self.train:
            d = np.array(d["notes"])
            ds = np.split(d, list(range(split_size,len(d),split_size)))
            for q in ds:
                if q[0] != self.hold_pitch and len(q) == split_size:
                    self.train_set.append(q)
            if i % 1000 == 0:
                print("processed: %d\tdivision: %d" %(i,len(self.train_set)))
            i = i + 1
        print("begin processing validate:")
        i = 0
        for d in self.validate:
            d = np.array(d["notes"])
            ds = np.split(d, list(range(split_size,len(d),split_size)))
            for q in ds:
                if q[0] != self.hold_pitch and len(q) == split_size:
                    self.validate_set.append(q)
            if i % 1000 == 0:
                print("processed: %d\tdivision: %d" %(i,len(self.validate_set)))
            i = i + 1
        print("begin processing test:")
        i = 0
        for d in self.test:
            d = np.array(d["notes"])
            ds = np.split(d, list(range(split_size,len(d),split_size)))
            for q in ds:
                if q[0] != self.hold_pitch and len(q) == split_size:
                    self.test_set.append(q)
            if i % 1000 == 0:
                print("processed: %d\tdivision: %d" %(i,len(self.test_set)))
            i = i + 1
        self.train_set = np.array(self.train_set)
        self.validate_set = np.array(self.validate_set)
        self.test_set = np.array(self.test_set)
        print("finished!")            
    def start_new_epoch(self, batch_size = 64):
        np.random.shuffle(self.train_set)
        np.random.shuffle(self.validate_set)
        np.random.shuffle(self.test_set)
        self.batch_idx = 0
        self.batch_size = batch_size
        self.train_batches = np.split(self.train_set,
            range(batch_size, self.train_set.shape[0] // batch_size * batch_size, batch_size))
        self.validate_batches = np.split(self.validate_set,
            range(batch_size, self.validate_set.shape[0] // batch_size * batch_size, batch_size))
        print("new epoch start!")
        return self.train_batches, self.validate_batches
    def convert_onehot(self, batch_data = None):
        vecs_out = []
        for d in batch_data:
            midi_vec = np.zeros((len(d),130))
            k = np.arange(len(d))
            midi_vec[k,d] = 1
            vecs_out.append(midi_vec)
        return np.array(vecs_out)
    def get_test(self):
        return self.test_set

    
    






            

