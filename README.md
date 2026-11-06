# TF2 Demo Analyser
A program used to analyse TF2 demos and detect cheaters. Forked from [nocrex's demo analyser](https://github.com/Nocrex/demo-analysis), which is based on [megascatterbomb's demo analysis template](https://github.com/MegaAntiCheat/analysis-template-rust). It features algorithms used to detect most of the detectable cheat features, a live analysis option to analyse the demo of an ongoing match, party chat warnings through RCON, TF2BD playerlist implementation and many other useful utility features. The algorithms have pretty self-explanatory names, for explanations on individual cheat features, check the relevant channel on discord.

The program is split into two separate builds: CLI (command line interface) and GUI (graphical user interface)

## **GUI**
GUI is the one that you are likely looking for. It includes all of the features needed to analyse and review demos with an easy to use interface.
1. Download the latest GUI build. It is recommended to keep the default settings. You may or may not want to disable the ML (machine learning) algorithms, more info about them can be found below.
2. Click **Open...**, pick a demo, then **Analyse**.
3. Flagged players are listed. Red = already marked, grey = deleted Steam account.
4. **Convert to replay** adds the demo to TF2's replays. Restart TF2 and find it by demo name or flagged SteamID.
5. **Copy detections** copies a formatted .md file, which you can easily paste in your report.

### **Live scanning**
Scans the demo TF2 is recording, by default on every death.
1. Add to `autoexec.cfg`:
```
ip 0.0.0.0
hostport 27015
rcon_password "password"
net_start
```
2. Add `-usercon` to TF2's Steam launch options.
3. Enter the same port and password in the analyser's RCON settings. Do not set the scan interval below 30 seconds.
4. Tick **Live**. The analyser should be able to find the current demo by itself.

## **CLI**
The CLI build offers the same detection algorithms as the GUI build, but uses the command line. It is most useful for **scanning demos in bulk**. To run a bulk scan on the default TF2 directory, open command line in the folder with `cli.exe` and run:
```
cli.exe --bulk --ignore-marked --ignore-deleted --replay-new
```
This scans your TF2 demos folder, skips marked and deleted players, creates replays for suspects and sorts demos into `sus` and `clean` subfolders. `--no-move` to leave demos in place. Use `--bulk "PATH"` to scan another folder if you do not use the default demo directory. To see all options and explanations for them: `cli.exe --help`

### **False flags**
Flags alone are not proof, so always check the demo yourself and, if possible, record video evidence. Isolated false flags are mostly removed by the sliding window filter, and several flags on one player in a short time span almost always mean a cheater. If unsure, post the report anyway; at worst it gets denied.
Most false-flag-prone: `fidoo/auto_backstab`, `fidoo/plain_aimbot`,`fidoo/sniperaim`

### **ML algorithms**
These algorithms use machine learning and were trained on ~5k demos from the database. They are very sensitive, so they flag players the standard algorithms miss, and are less prone to false flags. However, they are heavier and always run on the whole demo. They may also flag players and retract the flags later on. Due to this, I recommend not using them during Live scanning or setting the scanning interval to at least 120 seconds. `ml_triggerbot` and `ml_backtrack` are still prone to false flags and *may be improved in the future*, for now they are disabled by default.


### **Building from source**
You need to have cargo installed to build this program from source. 
There are several optional build features, it is recommended to keep all of them.
 - parallel - runs multiple workers at once to analyse demos faster
 - RCON - adds features that use RCON: party chat warnings and live scanning
 - TF2BD - TF2BD playerlist support, used to mark players with custom playerlist

To build the GUI with all features:
`cargo build --release --bin gui --features "gui rcon parallel"` 
To build the CLI:
`cargo build --release --bin cli --features "parallel"`
