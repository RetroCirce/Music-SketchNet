import pretty_midi
import numpy as np



min_step = 0.125 # 0.5 / 4 s
bad_re = []


chord_index = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
chord_group = 2
chord_size = 26
chord_check = ["C","Cm","C#","C#m","D","Dm","D#","D#m","E","Em","F","Fm","F#","F#m","G","Gm","G#","G#m","A","Am","A#", "A#m","B","Bm","NC"]
# major-0 minor-1 NC = 24
none_chord = 24
hold_chord = 25
rest_pitch = 129
hold_pitch = 128



chord_stage = 4
chord_order = [[0,4,7],[0,3,7]]

def gen_chord(chord_num):
	if chord_num == none_chord:
		return []
	pitch_pos = int(chord_num / 2)
	type_pos = int(chord_num % 2)
	re = []
	for i in range(3):
		re.append(chord_stage * 12 + chord_order[type_pos][i] + pitch_pos)
	return re


def innote(step, notes, pre_note=None):
    if (pre_note is not None) and \
        (np.around(step, 3) >= np.around(pre_note.start, 3)) and \
            (np.around(step, 3) < np.around(pre_note.end, 3)):
        return -2, pre_note
    for note in notes:
        if (np.around(step, 3) >= np.around(note.start, 3)) and \
                (np.around(step, 3) < np.around(note.end, 3)):
            return note.pitch, note
    return -1, None


def numpy_to_midi(sample_roll, output='sample/sample.mid'):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    t = 0
    for sample in sample_roll:
        for i in sample:
            if 'torch' in str(type(i)):
                pitch = int(i.max(0)[1])
            else:
                pitch = int(np.argmax(i))
            if pitch < 128:
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
                t += 1 / 8
                piano.notes.append(note)
            elif pitch == 128:
                if len(piano.notes) > 0:
                    note = piano.notes.pop()
                else:
                    p = np.random.randint(60, 72)
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(p),
                        start=0,
                        end=t
                    )
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end + 1 / 8)
                piano.notes.append(note)
                t += 1 / 8
            elif pitch == 129:
                t += 1 / 8
    music.instruments.append(piano)
    music.write(output)

def numpy_to_midi_with_condition(sample_roll,condition, output='sample/sample.mid'):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    chord_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    chord = pretty_midi.Instrument(program=chord_program)
    t = 0
    for sample in sample_roll:
        for i in sample:
            if 'torch' in str(type(i)):
                pitch = int(i.max(0)[1])
            else:
                pitch = int(np.argmax(i))
            if pitch < 128:
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
                t += 1 / 8
                piano.notes.append(note)
            elif pitch == 128:
                if len(piano.notes) > 0:
                    note = piano.notes.pop()
                else:
                    p = np.random.randint(60, 72)
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(p),
                        start=0,
                        end=t
                    )
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end + 1 / 8)
                piano.notes.append(note)
                t += 1 / 8
            elif pitch == 129:
                t += 1 / 8

    prev = none_chord
    total = 0
    timeline = 0
    for notet in condition:
        notet = notet[0]
        if notet == hold_chord:
            total = total + 1
        else:
            if prev == none_chord:
                prev = notet
                timeline = timeline + total * min_step
                total = 1
                continue
            s = gen_chord(prev)
            for o in s:
                note = pretty_midi.Note(velocity = 100, pitch = o,start = timeline, end = timeline + total * min_step )
                chord.notes.append(note)
            prev = notet
            timeline = timeline + total * min_step
            total = 1
    if prev != none_chord:
        s = gen_chord(prev)
        for o in s:
            note = pretty_midi.Note(velocity = 100, pitch = o,start = timeline, end = timeline + total * min_step )
            chord.notes.append(note)

    music.instruments.append(piano)
    music.instruments.append(chord)
    music.write(output)





def midi_to_numpy(filepath):
    data = []
    music = pretty_midi.PrettyMIDI(filepath)
    notes = music.instruments[0].notes
    beats = np.array(music.get_beats(), dtype=float)
    piece = np.zeros((beats.shape[0] * 4, 130)).astype(int)
    idx = 0
    for i in range(beats.shape[0] - 1):
        steps = np.linspace(beats[i], beats[i + 1], 4, False)
        pre_note = None
        for j in range(len(steps)):
            pitch, pre_note = innote(steps[j], notes, pre_note)
            piece[idx, pitch] = 1
            idx += 1
    margin = np.unique(np.diff(np.linspace(beats[-2], beats[-1], 4, False)))[0]
    for i in range(4):
        pitch, pre_note = innote(beats[-1] + margin * i, notes, pre_note)
        piece[idx, pitch] = 1
        idx += 1
    bars = np.split(piece, range(0, len(piece), 16), 0)
    data = []
    for i in range(len(bars) - 1):
        if bars[i].shape == bars[i + 1].shape:
            roll = np.vstack((bars[i], bars[i + 1]))
            fliped = flip_roll(roll)
            moved = []
            for r in fliped:
                moved += move_roll(r)
            data += moved
    return data


def flip_roll(roll):
    sample_roll = roll.copy()
    fliped = np.flip(sample_roll[:, :-2], -1)
    fliped = np.pad(fliped, ((0, 0), (0, 2)), 'constant')
    rest_where = np.where(sample_roll[:, -1] == 1)
    fliped[rest_where, -1] = 1
    rest_where = np.where(sample_roll[:, -2] == 1)
    fliped[rest_where, -2] = 1
    return [fliped, roll]


def rev_roll(sample_roll):
    rev = np.flip(sample_roll.copy(), 0)
    t = -1
    for i in range(rev.shape[0]):
        if (t == -1) and (rev[i, -2] == 1):
            t = i
        elif (t != -1) and (rev[i, -2] != 1):
            rev[t] = np.zeros(rev.shape[1])
            rev[t, np.argmax(rev[i])] = 1
            rev[i] = np.zeros(rev.shape[1])
            rev[i, -2] = 1
            t = -1
    return [rev, sample_roll]


def move_roll(sample_roll):
    idx = np.where(sample_roll[:, :-2] == 1)
    output = []
    if len(idx[1]) > 0:
        for p in range(0, 13):
            if (idx[1][0] >= 72):
                p_diff = idx[1] - p
            elif idx[1][0] < 60:
                p_diff = idx[1] + p
            else:
                p_diff = idx[1] + p - 6
            n_idx = (idx[0], p_diff)
            new_roll = np.zeros_like(sample_roll[:, :-2])
            new_roll[n_idx] = 1
            new_roll = np.hstack((new_roll, sample_roll[:, -2:]))
            output.append(new_roll)
    else:
        output.append(sample_roll)
    return output
