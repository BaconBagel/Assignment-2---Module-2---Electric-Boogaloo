from mido import MidiFile
import csv

mid = MidiFile('moonlight.mid')

all_notes = []
for y in range(90):
    all_notes.append([y])
    total_time = 0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.is_meta:
                print("meta")
            elif hasattr(msg,"note"):
                if int(msg.note) == y and msg.channel==0:
                    all_notes[y].append([total_time,msg])
                total_time += msg.time


count = 0
binaries = []
count = 0
for y in all_notes:
    print(y)
for x in range(90):
    if len(all_notes[x]) > 1:
        print(x)
        binaries.append([all_notes[x][0]])
        step_count = 0
        time_count = 0
        for n in range (1,len(all_notes[x])-1):
            print(all_notes[x][n][0], all_notes[x][n + 1][0], step_count)
            time_count = all_notes[x][n+1][0] - all_notes[x][n][0]
            print(time_count)
            print(all_notes[x][n][1],all_notes[x][n+1][1])
            if str(all_notes[x][n][1])[0:7] == "note_on":
                binaries[count].extend(["2", time_count])
                print("it do")
            print(str(all_notes[x][n][1])[0:8])
            if str(all_notes[x][n][1])[0:8] == "note_off" or str(all_notes[x][n][-2])[-1] == "0" :
                binaries[count].extend(["1",time_count])
                print("it dont")

        count += 1
from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
for z in binaries:
    if len(z)> 4:
        with open('moonlight.csv', 'a') as file2:
            writer = csv.writer(file2)
            writer.writerow(z)


for z in binaries:
    list_count = 0
    for p in range(1, int(len(z)/2)-1):
        p = 2*p
        if int(z[p+1]) == 2:
            track.append(Message('note_on', note=z[0], velocity=64, time=int(z[p])))
        else:
            print(z[p + 1])
            track.append(Message('note_off', note=z[0], velocity=127, time=int(z[p])))

mid.save('new_song.mid')


