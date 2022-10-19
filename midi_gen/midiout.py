from mido import MidiFile
import csv



all_notes = []
rows = []
with open("hut.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
from mido import Message, MidiFile, MidiTrack
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
for z in rows:
    if len(z[1]) > 1:
        print(z)
        note = int(z[0][1:3])
        z = z[1][1:-1].split(', ')
        print(note)
        for played in z:
            played = played[1]
            if played == "2":
                track.append(Message('note_on', note=note, velocity=64, time=200))
            else:
                track.append(Message('note_off', note=note, velocity=127, time=200))
mid.save('satie_emul.mid')


