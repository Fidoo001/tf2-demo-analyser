DEMO ANALYSIS - with built-in 3D demo player
=============================================

Finds suspicious players in Team Fortress 2 demos (.dem files) and lets you
watch any moment in 3D, without opening TF2.


WHAT YOU NEED
-------------
- Windows 10 or 11 (64-bit)
- Team Fortress 2 installed through Steam. The 3D player reads the maps,
  models and animations straight from your TF2 folder. Nothing from the game
  is included in this zip. TF2 does not need to be running.
- A graphics card with OpenGL 3.3 (any PC that runs TF2 is fine)


HOW TO START
------------
1. Unzip this folder anywhere.
2. Double-click "Demo Analysis.exe".
   Windows may say it "protected your PC" because the program isn't signed.
   Click "More info", then "Run anyway".
3. Click Open..., pick a demo (TF2 keeps them in
   Team Fortress 2\tf\demos), then Analyse.
4. Click "Watch demo" to open the 3D player, or the small play button next to
   any detection to jump straight to it.

GUIDE.md is the full user guide, with screenshots of every part of the
program. Open it in any Markdown viewer (VS Code, GitHub, Obsidian) or a
text editor.


3D PLAYER CONTROLS
------------------
Space          play / pause
Left / Right   skip 5 seconds (with Shift: 30 seconds)
,  .           step one frame back / forward
[  ]           slower / faster
T              all traces on / off
G              glow: Everyone -> Enemies -> Teammates -> Off
H              hitboxes on / off
A              aim lines on / off (when watching a player)
M              mute / unmute
C              first person / third person (when watching a player)
Esc            leave review mode

View menu      watch through any player's eyes (sniper scope included),
               or use the free camera
Third person   next to View (or C): watch from behind the player. Drag to
               swing the camera around them, scroll to zoom, double-click
               to go back behind them
Free camera    drag with the mouse to look, W A S D to move,
               Q / E down and up, Shift to go faster

Show... menu (top bar)
  Names              player names over heads
  Bullets            where hitscan shots went
  Projectile trails  rocket, pipe, sticky, flare and arrow trails
  Bullet impacts     a mark where each bullet hit
  Damage numbers     damage over the player who took it (only the watched
                     player's own hits when watching someone)
  Aim lines          where every player is looking, through walls too
  Aim graph          how fast the watched player's view turns, above the
                     timeline; an aimbot snap is one tall spike
  Aimbot FOV circle  when watching a flagged player: a circle around the
                     crosshair as big as their biggest flick onto a target,
                     which their aimbot reaches at least (also in review mode
                     and in player reports)
  Hitboxes           the boxes bullets are tested against (head in yellow)
  Backtrack ghosts   where the server puts enemies to judge the watched
                     player's shots: one round trip plus the interp delay
                     ago (set the interp in the menu)
Sound            volume; click it (or press M) to mute. You hear what the
                 recording player's game heard, voice chat included.
                 Sound plays at up to 2x speed; clips are saved silent.
Glow             team-coloured outlines through walls
FOV              field of view

Player list (right side)
  Shows class, health and ping (the scoreboard ping; green under 80 ms,
  amber to 150, red above). Right-click a player to:
    - watch through their eyes
    - review their detections (see Review mode below)
    - mark them in your TF2BD lists (turn TF2BD on in the main window)
    - export a report: one HTML page with their detections, kills,
      damage, accuracy and a ping chart, saved in the "reports" folder
    - open their Steam profile or copy their SteamID64

Review mode
  Pick a player from "Review..." above the Detections list, or right-click
  them in the player list. The player then plays each of their detections
  in turn: slow motion (0.25x) from their eyes up to the moment it was
  flagged, a short pause on it, then a little after, with aim lines and
  the aim graph on. A bar over the view says which check fired, what to
  look for on screen, and what the check measured in plain words (angles,
  distances, ticks, the target's name, the model's score). Previous,
  Replay and Next step between detections; "Go on by itself" moves on
  automatically. Exit review (or Esc) puts your settings back.

Save clip... (bottom bar)
  Saves 5, 10 or 20 seconds around the current moment, with everything
  on screen, into the "clips" folder. MP4 needs ffmpeg installed
  (https://ffmpeg.org); without it, clips are saved as GIF.

Updates
  When a newer build is released (https://github.com/wombo14/tf2-demo-analyser/
  releases), the main window shows "Update available" next to Open...
  Click it to download the new version, check it against its published
  checksum and restart. Your settings, reports and clips stay, and the
  version you had is kept next to it as "Demo Analysis (previous).exe" in
  case you want to go back.

  New builds bring in Fidoo001's changes to the detection checks: a robot
  on the GitHub repository checks his project every day, and his changes
  are released once they pass the tests. Until then, "N upstream updates"
  next to Open... shows that he has changed something.


FILES THE PROGRAM CREATES
-------------------------
params.json        your settings, saved next to the program
reported_ids.txt   a public list of known cheaters, downloaded the first time
                   it's needed
reports\           player reports you export from the 3D player
clips\             clips you save from the 3D player

Delete either one to start fresh.


COMMAND-LINE VERSION
--------------------
"demo-analysis-cli.exe" does the same analysis without a window.
Run it from a terminal with --help to see the options.


CREDITS AND LICENSE
-------------------
Demo Analysis is by Fidoo001 and contributors
(https://github.com/Fidoo001/tf2-demo-analyser), licensed under the
GNU GPL version 3 (LICENSE.md). This is a modified version. It adds:
  - the built-in 3D demo player (source/src/player)
  - the TF2BD community lists as default playerlists
It is shared under the same GPL version 3, and its full source is in the
"source" folder.
CHANGES.txt lists every change, with dates.

The optional TF2 Bot Detector lists (TF2BD menu) are downloaded from their
maintainers: the TF2BD Community Biglist and TF2BD Trusted.


SOURCE CODE
-----------
The "source" folder has everything needed to build it yourself:
  1. Install Rust from https://rustup.rs
  2. In the source folder run:
       cargo build --release --bin gui --features gui,rcon,tf2bd,parallel
     The program appears in target\release\gui.exe
