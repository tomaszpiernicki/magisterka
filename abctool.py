#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, string, re, optparse, os, tempfile, codecs, shutil

external_programs = {}
external_functionality = {}

external_programs['abc2ps'] = 'abcm2ps'
external_functionality['abc2ps'] = 'conversion to postscript'

external_programs['abc2abc'] = 'abc2abc'
external_functionality['abc2abc'] = 'external transposition'

external_programs['abc2xml'] = 'abc2xml'
external_functionality['abc2xml'] = 'conversion to musicXML'

external_programs['abc2midi'] = 'abc2midi'
external_functionality['abc2midi'] = 'conversion to midi'

external_programs['gv'] = 'gv'
external_functionality['gv'] = 'displaying postscript'

external_programs['gs'] = ['gs', 'gswin32c']
external_functionality['gs'] = 'converting to PDF'

external_programs['lpr'] = 'lpr'
external_functionality['lpr'] = 'printing'

external_programs['midiplayer'] = 'timidity'
external_functionality['midiplayer'] = 'playing back abc'

external_programs['editor'] = 'emacs'
external_functionality['editor'] = 'editing'

external_programs['file'] = 'file'
external_functionality['file'] = 'determining file type'

external_programs['iconv'] = 'iconv'
external_functionality['iconv'] = 'converting incoding'

external_programs['pdftk'] = 'pdftk'
external_functionality['pdftk'] = 'adding correct title to PDF files'

"""
latin = 'ISO-8859'

def get_encoding(file):
    result = os.popen('file "' + file + '"', 'r').readlines()
    try:
        encoding = result[0].split('\n')[0].split(':')[1].strip().split(' ')[0]
        return encoding
    except:
        pass

def change_encoding_to_utf8(file):
    tmpFile = tempfile.mktemp()
    encoding = get_encoding(file)
    if encoding == latin:
        print('in change encoding, file: ' + file)
        os.system('iconv -f latin1 -t utf8 ' + file + ' -o ' + tmpFile)
        shutil.move(tmpFile, file)
    #sys.exit()
    """     







class abctool:
    version = '0.4.17'
    date = '2019-10-01'
    
    manual = """
To view the list of options:
$ abctool.py

To edit an abc file (on my system it opens the source in emacs and
shows the postscript in gv)
$ abctool.py -e test.abc

To transpose the file "test.abc" a minor third up, convert it to 
postscript and view the result in gv:
$ abctool.py -t 3 test.abc

To do the same using abc2abc:
$ abctool.py -T 3 test.abc

To remove fingerings from the file "test.abc", translate chords to
danish and send the result to standard out:
$ abctool.py -f -d -s test.abc

To remove all chords from the file "test.abc", convert it to
postscript in landscape format and view the result in gv:
$ abctool.py -c -o "-l" test.abc

To make a songbook of all .abc-files in the current directory:
$ cat *.abc | abctool.py -r -

To generate PostScript, PDF and midi output from the file
"test.abc" with "m7b5" substituted with "ø" and "dim7" with "°":
$ abctool.py -j --ps --pdf test.abc

To listen to the file "test.abc":
$ abctool.py -p test.abc

To generate a pdf file (songbook.pdf) containing all songs recursively
in current directory in concert key, Bb and Eb, bypassing ~/.abctoolrc:
$ abctool.py --songbook -R --norc
"""





    changelog = """
--- Version 0.4.17 - 2019-10-01 ---
* --list_titles also prints the number, useful for use (afterwards) with -n

--- Version 0.4.16 - 2019-09-04 ---
* added --list_titles

--- Version 0.4.15 - 2019-02-25 ---
* better filename with --pdf and '-' in title

--- Version 0.4.14 - 2017-06-23 ---
* jazz conversions (-j) now also substitues maj7 with a triangle

--- Version 0.4.13 - 2014-08-29 ---
* filename with -n doesn't contain "," anymore

--- Version 0.4.12 - 2014-03-25 ---
* if single tune is selected with -n set outout filename from title when using --ps and --pdf

--- Version 0.4.11 - 2014-03-18 ---
* adding correct title (extracted from T:) to pdf files using pdftk

--- Version 0.4.10 - 2013-03-14 ---
* added -O for specifying outfile
* add total number of pages to ps or pdf file if possible, requires
  you to have something like this in your format file:
  footer 		"$T - $P"

--- Version 0.4.9 - 2012-09-23 ---
* generates blank postscript with -e on new file

--- Version 0.4.8 - 2012-04-18 ---
* -c broken with empty annotations
* -c was broken in lines *not* starting with chord

--- Version 0.4.7 - 2012-01-30 ---
* -c (remove chords) no longer removes annotations (chords
  beginning with "@") 

--- Version 0.4.6 - 2011-11-06 ---
* removed debug stuff that broke transposition

--- Version 0.4.5 - 2011-11-02 ---
* % comment in K: works

--- Version 0.4.4 - 2011-10-26 ---
* remove chords (-c) now works with stacked chords

--- Version 0.4.3 - 2011-08-30 ---
* Fixed another python3 bug

--- Version 0.4.2 - 2011-08-16 ---
* Reading of utf-8 and latin-1 files now work

--- Version 0.4.1 - 2011-08-12 ---
* Fixed some bad syntax with python 3

--- Version 0.4.0 - 2011-08-01 ---
* abctool.py now compatible with python 3

--- Version 0.3.27 - 2011-05-18 ---
* -C not working together with -d

--- Version 0.3.26 - 2011-03-23 ---
* -a no longer removes annotations (guitarchords starting with @):
* added -@ to remove annotations

--- Version 0.3.25 - 2011-01-24 ---
* -t messed up in-tune T:

--- Version 0.3.24 - 2010-10-26 ---
* -D broken with empty line in input

--- Version 0.3.23 - 2010-10-24 ---
* removed debug print of entire abc with -D

--- Version 0.3.22 - 2010-10-12 ---
* fixed transposition bug when [K:] in individual voice
* -d should only process after header

--- Version 0.3.21 - 2010-09-20 ---
* fixed transposition bug with keys written as modes

--- Version 0.3.20 - 2010-07-08 ---
* initial support for multiple platforms:
  gs (linux) and gswin32c.exe (windows), thanks to Nathan Gourley

--- Version 0.3.19 - 2010-04-16 ---
* --songbook sorts files

--- Version 0.3.18 - 2010-01-20 ---
* added -P for printing via lpr

--- Version 0.3.17 - 2010-01-18 ---
* transposition would mess up inline [V and [I
* -s disables -d (danish chords) and -j (jazz chords)

--- Version 0.3.16 - 2010-01-15 ---
* fixed bug with -n and -X, 10 would select 1 and 0

--- Version 0.3.15 - 2010-01-15 ---
* added -X, tune selection based on X:field
* cleaned up the code for handling -n

--- Version 0.3.14 - 2010-01-14 ---
* fixed bug with stranger combinations of[V:]

--- Version 0.3.13 - 2010-01-14 ---
* -n didn't work with other processing
* added support for inline voice declarations ([V:])

--- Version 0.3.12 - 2010-01-13 ---
* fixed bugs with -V

--- Version 0.3.11 - 2009-11-16 ---
* added -D (remove all decorations)

--- Version 0.3.10 - 2009-10-09 ---
* chords beginning with h-z, æ, ø, å, H-Z, Æ, Ø, Å are also regarded as
  annotations, and thus not transposed

--- Version 0.3.9 - 2009-10-07 ---
* danishChords disregards annotations (chords beginning with @)

--- Version 0.3.8 - 2009-10-06 ---
* text inside %%begintext and %%endtext is not transposed

--- Version 0.3.7 - 2009-04-22 ---
* timestamp of abc file is kept with viewing and conversion to ps and pdf

--- Version 0.3.6 - 2009-01-06 ---
* added --title, adds another title (T:) below existing titles

--- Version 0.3.5 - 2009-01-05 ---
* fixed bug with multiple tunes (X:n)

--- Version 0.3.4 - 2008-09-28 ---
* fixed -n
* handles non-tune lines better
* fixed transposition bug with other scale types than major
* added --list_voices -V (keep only voices)
* fixed --songbook, --pdf and --ps
* (re)added replacement of strings (-r)

--- Version 0.3.3 - 2008-09-28 ---
* -e works again (with abc_options)

--- Version 0.3.2 - 2008-09-28 ---
* added -o (abcm2ps options)

--- Version 0.3.1 - 2008-09-28 ---
* compatible with python 2.4

--- Version 0.3.0 - 2008-09-27 ---
* major refactoring and code cleanup
* now works with more than one input file
* removed dependency of texexec and touch
* transposition works with upto 9 accidentals

--- Version 0.2.12 - 2007-06-20 ---
* various small fixes

--- Version 0.2.11 - 2007-01-16 ---
* Added -L (extract lyrics)

--- Version 0.2.10 - 2006-06-15 ---
* fixed major bug that resulted in keys (with mode different from '')
  would only use white key for root (C#m would become Cm)

--- Version 0.2.9 - 2006-06-09 ---
* K:Hp and K:HP now transposes by uncommenting K:HP and adding K:Amix
  before regular transposition 

--- Version 0.2.8 - 2006-06-09 ---
* transposition now works with church modes, written in
  upper/lower/mixed and/or abbreviated to first three letters
* code cleanup: more centralized setup of scales
* --songbook doesn't try to remove logfiles, which makes it exit
  cleanly 

--- Version 0.2.7 - 8/6-2006 ---
* chords with double altered roots wasn't transposed
* added -C switch to reduce accidentals in chords, for instance Cb
  becomes B and C## becomes D
* help message is maximum 80 chars wide
* --songbook [-R] now reacts to -j, -C, -d and -g
* key type maj now works

--- Version 0.2.6 - 7/6-2006 ---
* notes with leading _ or ^ after closing " in guitar chords were not
  transposed
* now treats 'K:' (with missing key) as 'K:none'

--- Version 0.2.5 - 6/6-2006 ---
* added more elaborate diagnostics with --check, report which part of
  the program that will work/fail

--- Version 0.2.4 - 6/6-2006 ---
* added a --check switch, which check to see if (potentionally)
  needed external programs are present on the system

--- Version 0.2.3 - 5/6-2006 ---
* annotations (guitar chords starting with one of <, >, _, ^, or @)
  are left alone during transposition

--- Version 0.2.2 - 31/5-2006 ---
* non-standard octaves (like C' and c,) are handled in transposition
* keys are now stopped at the first space, this should be changed to
  actually handling modes and exp, but it's better than throwing an
  exception 
* decorations using + (ex: +fermata+) are now left alone by the
  tranposition 

--- Version 0.2.1 - 30/5-2006 ---
* K:none was broken, at least with transposition > 12 or < -12
* chords using [] was not transposed, since they were considered
  in-tune control fields (like [M:] and [K:]. now only [K and [M are
  left alone, others are transposed

--- Version 0.2.0 - 28/5-2006 ---
* implemented correct transposition. This is now the default and
  called with -t, whereas -T is using abc2abc

--- Version 0.1.4 - 25/7-2004 ---
* added --songbook and --songbook -R for generating songbook.pdf
  containing 
  all files in current directory in the concert key, Bb and Eb
* added --norc for not using ~/.abctoolrc
* added --nostdargs for not using standard_args from ~/.abctoolrc

--- Version 0.1.3 - 26/3-2004 ---
* added -e for editing
* added -w, -W and -l for removing lyrics

--- Version 0.1.2 - 21/3-2004 ---
* removed the functions standardize and de_standardize (not needed by
  newer versions of abc2abc)
* added --midi output and -p option (play using midi and a midifile
  player)
* added support for ~/.abctoolrc

--- Version 0.1.1 - 20/3-2004 ---
* added option -r for reshufling songs, giving them successive tune
  numbers (X: fields)
* added encoding (latin-1)
* added option -j that translates jazz chords (m7b5 becomes ø, dim7
  becomes °)
* made -f and -c more robust (disregards header)
* added --ps and --pdf for generating PostScript and PDF
* changed the formatting of -h to be grouped by functionallity instead
  of alphabetically

--- Version 0.1.0 - 15/3-2004 ---
Initial release

Features:

the following options are supported:
-       read from stdin
-c      remove chords
-d      translate to danish chords (B becomes H), same as -g
-f      remove fingerings
-g      translate to german chords (B becomes H), same as -d
-h      show this help
-o O    call abc2ps wit additional options O
-s      send output to stdout
-t X    transpose by X halfsteps
-v      show version

abc2ps (actually abcm2ps) is used for generating PostScript, gv is
used to display PostScript and abc2abc is used to transpose
"""

    modTime = 0
    files = []
    abcs = []

    rcFile = os.path.expanduser('~/.abctoolrc')

    def __init__(self):
        self.externalProgramsCandidates()

    def printChangelog(self):
        print(self.changelog)
        sys.exit()

    def printManual(self):
        print(self.manual)
        sys.exit()

    def writeFile(self,string,file):
        f = open(file,'w')
        f.write(string)
        f.close()
        
    def reencode(self,string):
        print('in reencode')
        try:
            tmp = string.decode('utf-8')
        except:
            pass
        else:
            string = tmp.encode('latin-1')
        return string
        # unused...
        print(self.options.encoding)
        if self.options.encoding == 'latin-1':
            try:
                tmp = string.decode('utf-8')
            except:
                print('1pass:')
                pass
            else:
                print('1else:')
                string = tmp.encode('latin-1')
        elif self.options.encoding == 'utf-8':
            try:
                tmp = string.encode('utf-8')
            except:
                print('2pass:')
                pass
            else:
                print('2else:')
                string = tmp
        
        return string
            
    def readFile(self, file):
        nonTuneLines = ''
        try:
            f = open(file,'r')
            filecontent = f.read()
            f.close()
        except:
            filecontent = codecs.open(file, "r", "latin-1" ).read()


        
        if os.stat(file).st_mtime > self.modTime:
            self.modTime = os.stat(file).st_mtime
        # modTime FIXME
        # must set mtime of tmpfile to this later...
        #sys.exit()


        lines = filecontent.split('\n')
        lines.append('')
        inTune = False
        tune = ''
        nonTune = ''
        for line in lines:
            if line[:2] == 'X:':
                tune = ''
                inTune = True
                
            if inTune:
                if line.strip() == '':
                    inTune = False
                    if nonTune == '':
                        abcTune = abctune(tune)
                    else:
                        abcTune = abctune(nonTune + '\n' + tune)
                    abcTune.abcOptions = self.options.abcOptions
                    self.abcs.append(abcTune)
                else:
                    tune = tune + line + '\n'
            else:
                if line.strip() != '':
                    nonTune = nonTune + line + '\n'
                


    def readFileOld(self, file):
        nonTuneLines = ''
        f = open(file,'r')
        filecontent = f.read()
        f.close()
        tunes = filecontent.split("\n\n")
        for tune in tunes:
            print('----tune:')
            print(tune)
            print('-----tuneEOF')
            print()
            
            tune = self.reencode(tune)
            if  self.isTune(tune):
                sys.exit()
                abcTune = abctune(tune)
                abcTune.abcOptions = self.options.abcOptions
                self.abcs.append(abcTune)
            else:
                nonTuneLines = nonTuneLines + tune + '\n'
                print(nonTuneLines)

    def parseRcFile(self):
        try:
            args = []
            f = open(self.rcFile,'r')
            content = f.read()
            f.close()
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 0 and line[0] != '#':
                    splits = line.split("'")
                    i = 0
                    for split in splits:
                        if i%2 == 0:
                            for arg in split.split(' '):
                                if len(arg) > 0:
                                    args.append(arg.strip())
                        else:
                            args.append(split.strip())
                        i = i + 1
            args.reverse()
            for arg in args:
                sys.argv.insert(1,arg)
                

        except:
            pass

        

    def parseOptions(self):
        # show help if no input + if -?
        if len(sys.argv) == 1:
            sys.argv.append('-h')
        elif '-?' in sys.argv:
            sys.argv.remove('-?')
            sys.argv.append('-h')  

            
        self.parseRcFile()

        parser = optparse.OptionParser()

        # editing
        editGroup = optparse.OptionGroup(parser, 'Editing','')
        editGroup.add_option('-e', '', action='store_true', dest='edit', help='edit in external editor and view in gv', default=False)

        # selection
        selectGroup = optparse.OptionGroup(parser, 'Tune selection','')
        selectGroup.add_option('-X', '', dest='XtuneNumber', help='process tune(s) with matching X: field, comma seperated list', default=None)
        selectGroup.add_option('-n', '', dest='tuneNumber', help='process only selected tune(s) (comma seperated list), first tune in file is 1, if -X also present -n is disregarded', default=None)
        selectGroup.add_option('-V', '', dest='keepVoices', help='extract only voices (comma seperated list)', default='')
        selectGroup.add_option('', '--list_voices', action='store_true', dest='listVoices', help='list voices (V:) found in the abc', default=False)
        selectGroup.add_option('', '--list_titles', action='store_true', dest='listTitles', help='list titles (T:) found in the abc', default=False)


        # processing
        processGroup = optparse.OptionGroup(parser, 'Processing','')
        processGroup.add_option('-t', '', type='int', dest='transpose', help='transpose by TRANSPOSE halfsteps', default=0)
        processGroup.add_option('-T', '', type='int', dest='externalTranspose', help='transpose by TRANSPOSE halfsteps using abc2abc', default=0)
        processGroup.add_option('-A', '', action='store_true', dest='explicitAccidentals', help='explicit accidentals', default=False)
#        processGroup.add_option('-j', '', action='store_true', dest='jazzChords', help='translate jazz chords'.decode('utf-8'), default=False)
        processGroup.add_option('-j', '', action='store_true', dest='jazzChords', help='translate jazz chords', default=False)
        processGroup.add_option('-d', '', action='store_true', dest='danishChords', help='translate to danish chords (B becomes H)', default=False)


        processGroup.add_option('-c', '', action="store_true", dest='removeChords', help='remove chords', default=False)
        processGroup.add_option('-@', '', action="store_true", dest='removeAnnotations', help='remove annotations', default=False)
        processGroup.add_option('-f', '', action="store_true", dest='removeFingerings', help='remove fingerings', default=False)
        processGroup.add_option('-D', '', action="store_true", dest='removeDecorations', help='remove all decorations', default=False)

        processGroup.add_option('-w', '', action="store_true", dest='removeLyrics', help='remove lyrics (w:)', default=False)
        processGroup.add_option('-W', '', action="store_true", dest='removeExtraLyrics', help='remove lyrics (W:)', default=False)
        processGroup.add_option('-l', '', action="store_true", dest='removeAllLyrics', help='remove all lyrics (w: and W:)', default=False)
        processGroup.add_option('-L', '', action="store_true", dest='extractLyrics', help='extract lyrics (w: and W:)', default=False)


        processGroup.add_option('-C', '', action="store_true", dest='whiteChordRoots', help='make chord roots white keys (Fb becomes E)', default=False)
        processGroup.add_option('','--title', type="string", dest='addTitle', help='Add title (T:) below original title', default='')



        processGroup.add_option('-r', '', dest='replaceStrings', help='replace strings, comma seperated list', default='')


        # input/output
        outputGroup = optparse.OptionGroup(parser, 'Input/output','')



#        outputGroup.add_option('','--defaultabc', type="string", dest='defaultabc', help='default abc in new files', default='')
        outputGroup.add_option('-i', '', action="store_true", dest='stdin', help='read from stdin', default=False)
        outputGroup.add_option('-s', '', action="store_true", dest='stdout', help='send to stdout', default=False)
        outputGroup.add_option('', '--pdf', action="store_true", dest='pdf', help='convert to pdf', default=False)
        outputGroup.add_option('', '--ps', action="store_true", dest='ps', help='convert to PostScript', default=False)
#        outputGroup.add_option('', '--chord', action="store_true", dest='chord', help='convert to ChordPro', default=False)
        outputGroup.add_option('', '--songbook', action="store_true", dest='generateSongbook', help='generate songbook.pdf containing all .abc files in current directory in concert key and transposed for Bb and Eb instruments. If -R is present do it recursively', default=False)
        outputGroup.add_option('-R', '', action="store_true", dest='recursive', help='look for files recursively', default=False)
        outputGroup.add_option('', '--midi', action='store_true', dest='generateMidi', help='generate midi', default=False)
        outputGroup.add_option('-p', '', action='store_true', dest='playMidi', help='play', default=False)
        outputGroup.add_option('-P', '', action='store_true', dest='printFile', help='print', default=False)

        outputGroup.add_option('-a', '', action='store_true', dest='absPath', help='place output files in absolute paths (and not in current directory)', default=False)
        outputGroup.add_option('-O', '', dest='outFile', help='output to file (name and location)', default='')
        #outputGroup.add_option('', '--enc', dest='encoding', help='output encoding', default='latin-1')

        # options
        optionGroup = optparse.OptionGroup(parser, 'Options','')
        optionGroup.add_option('--abc_options', '-o', dest='abcOptions', help='call abc(m)2ps with these options', default='')
        optionGroup.add_option('--gv_options', '', dest='gvOptions', help='call gv with these options', default='')
        
        # diagnostics
        diaGroup = optparse.OptionGroup(parser, 'Dianostics','')
        diaGroup.add_option('', '--check', action="store_true", dest='checkExternalPrograms', help='check if certain external programs needed for some tasks are present', default=False)
        diaGroup.add_option('', '--manual', action="store_true", dest='manual', help='view manual', default=False)
        diaGroup.add_option('', '--changelog', action="store_true", dest='changelog', help='view changelog', default=False)
        diaGroup.add_option('-v', '--version', action="store_true", dest='printVersion', help='view version', default=False)
        

        parser.add_option_group(editGroup)
        parser.add_option_group(selectGroup)
        parser.add_option_group(processGroup)
        parser.add_option_group(outputGroup)
        parser.add_option_group(optionGroup)
        parser.add_option_group(diaGroup)

        (options, args) = parser.parse_args()
        self.options = options
        self.args = args

        if self.options.stdout:
            self.options.danishChords = False
            self.options.jazzChords = False
        
        if self.options.changelog:
            self.printChangelog()
            sys.exit()
        elif self.options.checkExternalPrograms:
            self.checkExternalPrograms()
            sys.exit()
        elif self.options.printVersion:
            self.printVersion()
            sys.exit()
        elif self.options.manual:
            self.printManual()
            sys.exit()
        elif self.options.generateSongbook:
            self.generateSongbook()
            sys.exit()
            
        if not '-h' in sys.argv:
            if len(args) == 0 and not '-i' in sys.argv:
                print('missing file')
                sys.exit()
            elif options.stdin or (len(args) > 0 and args[0] == '-'):
                infile = 'stdin'
                line = sys.stdin.read()
                abcTmpFile = tempfile.mktemp()
                f = open(abcTmpFile,'w')
                f.write(line)
                f.close()
                self.readFile(abcTmpFile)
                os.remove(abcTmpFile)

            for file in args:
                if os.path.exists(file):
                    self.readFile(file)
                else:
                    continue
                    print('file "' + file +'" doesnt exist')
                    





    def process(self):
        if self.options.edit:
            self.edit()
            sys.exit()

        if self.options.XtuneNumber:
            self.XtuneNumberKeep(self.options.XtuneNumber)
        elif self.options.tuneNumber:
            self.tuneNumberKeep(self.options.tuneNumber)

        if self.options.listVoices:
            self.listVoices()
            sys.exit()
        if self.options.listTitles:
            self.listTitles()
            sys.exit()
        if self.options.keepVoices:
            self.keepVoices()
        if self.options.removeLyrics:
            self.removeLyrics()
        if self.options.replaceStrings:
            self.replaceStrings()
        if self.options.removeExtraLyrics:
            self.removeExtraLyrics()
        if self.options.removeAllLyrics:
            self.removeLyrics()
            self.removeExtraLyrics()
        if self.options.extractLyrics:
            self.extractLyrics()
            sys.exit()
        if self.options.addTitle:
            self.addTitle()

        if self.options.transpose != 0:
            self.transpose()
        elif self.options.externalTranspose != 0:
            self.externalTranspose()

        if self.options.removeChords:
            self.removeChords()
        if self.options.removeAnnotations:
            self.removeAnnotations()
        if self.options.removeFingerings:
            self.removeFingerings()
        if self.options.removeDecorations:
            self.removeDecorations()
        if self.options.jazzChords:
            self.jazzChords()
        if self.options.whiteChordRoots:
            self.whiteChordRoots()
        if self.options.danishChords:
            self.danishChords()
        if self.options.explicitAccidentals:
            self.explicitAccidentals()
            
        if self.options.stdout:
            self.printAbc()
        #elif self.options.chord:
        #    self.generateChord()
        elif self.options.generateMidi:
            self.generateMidi()
            sys.exit()
        elif self.options.playMidi:
            self.playMidi()
        elif self.options.pdf:
            self.generatePdf()
        elif self.options.ps:
            self.generatePs()
        elif self.options.printFile:
            self.printFile()
        else:
            self.view()


    def printVersion(self):
        print(self.version + ', ' + self.date)



    def touch(self, file):
        if not os.path.exists(file):
            f = open(file,"w")
            f.write('')
            f.close() 
        
    def searchPath(self,cmdname, path = None):
        if path is None:
            path = os.environ["PATH"]

        if os.name in ["nt", "os2"]:
            short = [cmdname + "." + ext for ext in ["exe","com","bat"]]
        else:
            short = [cmdname]

        for scmd in short:
            for dir in path.split(os.pathsep):
                fcmd = os.path.abspath(os.path.join(dir,scmd))
                if os.path.isfile(fcmd):
                    return fcmd
        return None


    def externalProgramsCandidates(self):
        for program in external_programs:
            if type(external_programs[program]) == list:
                check_list = external_programs[program]
                external_programs[program] = None
                for candidate in check_list:
                    path = self.searchPath(candidate)
                    if path != None:
                        external_programs[program] = candidate
                        break

    def getCandidate(self,program):
        result = None
        if type(external_programs[program]) == list:
            for candidate in check_list:
                path = self.searchPath(candidate)
                if path != None:
                    result = candidate
                    break
            result = None
        else:
            result = external_programs[program]
        path = self.searchPath(result)
        if path != None:
            return result
        else:
            print("couldn't find executable for " + program + ', exiting...')
            sys.exit()
        
        

    def checkExternalPrograms(self):
        ok = 'ok    '
        error = 'error '
        for program in external_programs:
            try:
                test = external_functionality[program]
            except:
                test = ''
            else:
                test = 'ok'
            path = self.searchPath(external_programs[program])
            if path != None:
                if test == 'ok':
                    print(ok +'found ' + program + ' (used for ' + external_functionality[program] + ') in ' + path)
                else:
                    print(ok +'found ' + program + ' in ' + path)
                
            else:
                if test == 'ok':
                    print(error + 'couldn\'t find ' + program + ', ' + external_functionality[program] + ' won\'t work...')
                else:
                    print(error + 'couldn\'t find ' + program + ', some part(s) of ' + programname + ' won\'t work...')
                


    def generateSongbook(self):
        options = self.options.abcOptions
        files = []
        if '-R' in sys.argv:
            for root, dirs, allFiles in os.walk('.'):
                for file in allFiles:
                    files.append(root + '/' + file)
        else:
            for file in os.listdir('.'):
                if os.path.splitext(file)[1] == '.abc':
                    files.append(file)
        files.sort()
        for file in files:
            if os.path.exists(file):
                    self.readFile(file)
            else:
                continue
                print('file "' + file +'" doesnt exist')


        transpose = 0
        runs = [[0,'songbook.pdf'],[9,'songbook_eb.pdf'],[14,'songbook_bb.pdf']]
        for run in runs:
            transpose = run[0] - transpose
            abcTmpFile = tempfile.mktemp()
            psTmpFile = tempfile.mktemp()
            pdfFileName = run[1]
            if transpose != 0:
                for i in range(0, len(self.abcs)):
                    self.abcs[i].transpose(transpose)
            self.writeFile(self.getAbc(),abcTmpFile)
            self.abc2ps(abcTmpFile + ' -O ' + psTmpFile + '  ' + options)
            self.addTotalNbPages(psTmpFile)
            self.ps2pdf(psTmpFile,pdfFileName)
        os.remove(abcTmpFile)
        os.remove(psTmpFile)




    def generateMidi(self):
        i = 0
        for abc in self.abcs:
            basename = self.args[i]
            if self.options.outFile:
                basename = self.options.outFile
            elif not self.options.absPath:
                basename = os.path.basename(basename)
            basename = os.path.splitext(basename)[0]
            abc.generateMidi(basename + '.mid')
            i = i + 1
        sys.exit()

    def playMidi(self):
        for abc in self.abcs:
            abc.playMidi()
        sys.exit()



    def getOutBasename(self):
        basename = self.args[0]
        if self.options.outFile:
            basename = self.options.outFile
        elif not self.options.absPath:
            basename = os.path.basename(basename)
        basename = os.path.splitext(basename)[0]
        # if single tune with -n get basename from title
        if self.options.tuneNumber and not ',' in self.options.tuneNumber:
            title = self.getTitle()
            basename = title.lower().replace(' ','_')
        basename = basename.replace("'",'')
        return basename

    def ps2pdf(self,ps,pdf):
        command = self.getCandidate('gs')
        os.system(command + ' -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sPAPERSIZE=a4 -sOUTPUTFILE=' + pdf + ' ' + ps)


    def abc2ps(self,options):
        command = self.getCandidate('abc2ps')
        os.system(command + ' ' + options)


    def gv(self, options):
        command = self.getCandidate('gv')
        os.system(command + ' ' + options)


    def printFile(self):
        command = self.getCandidate('lpr')
        psFileName = self.getOutBasename() + '.ps'
        self.generatePs()
        os.system(command + ' < ' + psFileName)
        os.remove(psFileName)

    def getTitle(self):
        title = []
        abc = self.getAbc()
        
        for line in abc.split('\n'):
            line = line.strip()
            if line[:2] == 'T:':
                title.append(line[2:])
                

        title = '_'.join(title)
        return title

    def generateChord(self):
        def prepare_words(words):
            words = words.replace('w:','')
            words = words.replace(' ',' #')
            words = words.replace('*',' ')
            words = words.replace('-','#')
            words = words.replace('_','#')
            words = words.split('#')
            return words
        
        def strip_music(music):
            music = music.replace('z','')
            music = music.replace('Z','')
            music = music.replace('x','')
            music = music.replace('X','')
            music = music.replace(',','')
            music = music.replace("'",'')
            music = music.replace('/','')
            music = music.replace('-','')
            music = music.replace('(','')
            music = music.replace(')','')
            music = music.replace('|','')
            music = music.replace(']','')
            music = music.replace('[','')
            music = music.replace(':','')
            music = music.replace('2','')
            music = music.replace('4','')
            music = music.replace('3','')
            music = music.replace(' ','')
            music = music.replace(' ','')
            music = music.replace(' ','')
            music = music.replace(' ','')
            music = music.replace(' ','')
            music = music.replace(' ','')
            music = music.replace(' ','')
            music = music.strip()
            return music
            

        def prepare_music(music):
            #print(music)
            music = music.split('"')
            i = 0
            for i in range(0,len(music),2):
                music[i] = strip_music(music[i])
                #print(music[i])
            return music

        def string2list(string):
            result = []
            for i in range(0,len(string)):
                result.append(string[i])
            return result

        def implode_lines(music, words):
            words = prepare_words(words)
            music = prepare_music(music)
            result = []
            #print(music)
            #print(words)

            j_note = 0
            j_note_group = 0

            for i in range(0,len(music),2):
                music[i] = string2list(music[i])

            for i_word in range(0,len(words)):
                word = words[i_word]
                if j_note >= len(music[j_note_group]):
                    j_note_group = j_note_group + 2
                    j_note = 0
                #print('replaceing j_note:' + str(j_note) + ', j_note_group:' + str(j_note_group))
                music[j_note_group][j_note] = word
                j_note = j_note + 1
            #print(music)
            for i in range(0, len(music)):
                if type(music[i]).__name__ == 'list':
                    result.append(''.join(music[i]))
                elif type(music[i]).__name__ == 'str':
                    result.append('[' + music[i] + ']')
            result = ''.join(result)
            #print(result)
            return result

        chord = []
        past_title = False
        past_header = False
        title = []
        subtitle = []
        chordFileName = self.getOutBasename() + '.cho'
        abc = self.getAbc()
        lines = abc.split('\n')
        abc = []
        for line in lines:
            if past_header:
                dummy = ['%','P','V','W'].count(line[:1])
                if dummy == 0 and len(line.strip()) > 0:
                    abc.append(line)
            elif line[:2] == 'T:':
                if not past_title:
                    title.append(line[2:])
                    past_title = True
                else:
                    subtitle.append(line[2:])
            
            elif line[:2] == 'C:':
                subtitle.append(line[2:])
            elif line[:2] == 'K:':
                past_header = True
        i = 0
        for i in range(0,len(abc)):
            line = abc[i]
            #print(abc[i])
            if abc[i][:2] == 'w:':
                chord.append(implode_lines(abc[i-1], abc[i]))

        chord = '\n'.join(chord)
        print(chord)


    def addTotalNbPages(self, file):
        verbose = False
        nb_pages = '0'
        f = open(file,'r')
        filecontent = f.read()
        f.close()
        lines = filecontent.split('\n')
        # get nb_pages
        for line in lines:
            if line[:9] == '%%Pages: ':
                if line[9:].isdigit():
                    nb_pages = line[9:]
        
        if nb_pages == '0':
            if verbose:
                print('couldn\'t find nb_pages in file: ' + file + ', skipping...')
            return
    
        # get title
        title = ''
        for line in lines:
            if line[:9] == '% --- 2 (':
                if verbose:
                    print('found multiple titles in file ' + file + ', skipping...')
                return
            if line[:9] == '% --- 1 (':
                title = line[9:].split(')')[0]
        if not title:
            if verbose:
                print('no title found in ' + file)
            return

        # add nb_pages after title in footer
        new_lines = []
        modified = False
        for line in lines:
            if title + ' - ' in line and nb_pages == '1':
                line = re.sub(title + ' .*\)',title + ')',line)
                modified = True
            elif title + ' - ' in line and not '/' in line:
                line = line.replace(')','/' + nb_pages + ')')
                modified = True
            new_lines.append(line)
        if not modified:
            if verbose:
                print('file ' + file + ' already has nb pages added (or page number removed), skipping...')
        else:
            filecontent = '\n'.join(new_lines)
            f = open(file,'w')
            f.write(filecontent)
            f.close()


    def generatePs(self):
        options = self.options.abcOptions
        abcTmpFile = tempfile.mktemp()
        psFileName = self.getOutBasename() + '.ps'
        self.writeFile(self.getAbc(),abcTmpFile)
        os.utime(abcTmpFile,(self.modTime,self.modTime))
        self.abc2ps(abcTmpFile + ' -O ' + psFileName + '  ' + options)
        self.addTotalNbPages(psFileName)
        #print(self.getNbPages())
        #sys.exit()
        os.remove(abcTmpFile)


    def file2string(self,file):
        try:
            f = open(file,'r')
            filecontent = f.read()
            f.close()
        except:
            filecontent = codecs.open(file, "r", "latin-1" ).read()
        return filecontent

    def string2file(self,string,file):
        f = open(file,'w')
        f.write(string)
        f.close()

    def addTitleToPdf(self,pdfFileName):
        path = self.searchPath(external_programs['pdftk'])
        if path == None:
            return
        title = self.getTitle()
        infoTmpFile = tempfile.mktemp()
        pdfTmpFile = tempfile.mktemp()
        os.system('pdftk "' + pdfFileName + '" dump_data_utf8 output ' + infoTmpFile)
        newInfo = []
        titleIsNext = False
        info = self.file2string(infoTmpFile).split('\n')
        for line in info:
            if titleIsNext:
                line = line.split(':')
                line[1] = ' ' + title
                line = ':'.join(line)
                newInfo.append(line)
                titleIsNext = False
            else:
                newInfo.append(line)
            if 'InfoKey:' in line and 'Title' in line:
                titleIsNext = True


        newInfo = '\n'.join(newInfo)
        self.string2file(newInfo,infoTmpFile)
        os.system('pdftk "' + pdfFileName +'" update_info_utf8 ' + infoTmpFile + ' output ' + pdfTmpFile)
        shutil.move(pdfTmpFile,pdfFileName)
        os.remove(infoTmpFile)
        
                
    def generatePdf(self):
        options = self.options.abcOptions
        abcTmpFile = tempfile.mktemp()
        psTmpFile = tempfile.mktemp()
        pdfFileName = self.getOutBasename() + '.pdf'
        pdfFileName = pdfFileName.replace('_-_','-')
        self.writeFile(self.getAbc(),abcTmpFile)
        os.utime(abcTmpFile,(self.modTime,self.modTime))
        self.abc2ps(abcTmpFile + ' -O ' + psTmpFile + '  ' + options)
        self.addTotalNbPages(psTmpFile)
        self.ps2pdf(psTmpFile,pdfFileName)
        self.addTitleToPdf(pdfFileName)
        os.remove(abcTmpFile)
        os.remove(psTmpFile)

        
    def edit(self):
        psTmpFile = '/tmp/abc.ps'
        editfile = self.getCandidate('editor') + ' %f'
        files = []
        for file in self.args:
            if not os.path.isfile(file):
                self.touch(file)
                os.system('echo %! > ' + psTmpFile)
            files.append(file)
        
        # view
        options = self.options.abcOptions
        gv_view_options = self.options.gvOptions

        self.abc2ps(files[0] + ' -O ' + psTmpFile + '  ' + options)
        self.addTotalNbPages(psTmpFile)
        self.gv(gv_view_options + ' ' + psTmpFile + '&')

        # edit
        files.reverse()
        files = ' '.join(files)

        calledit = editfile.replace('%f',files) + ' &'
        os.system(calledit)

        sys.exit()

    def getNbPages(self, file):
        f = open(file,'r')
        filecontent = f.read()
        f.close()
        lines = filecontent.split('\n')
        for line in lines:
            if line[:7] == '%%Pages':
                print(line)
        return 10

    def view(self):
        options = self.options.abcOptions
        gv_view_options = self.options.gvOptions
        abcTmpFile = tempfile.mktemp()
        psTmpFile = tempfile.mktemp()
        self.writeFile(self.getAbc(),abcTmpFile)
        os.utime(abcTmpFile,(self.modTime,self.modTime))
        self.abc2ps(abcTmpFile + ' -O ' + psTmpFile + '  ' + options)
        self.addTotalNbPages(psTmpFile)
        #print(self.getNbPages(psTmpFile))
        self.gv(gv_view_options + ' ' + psTmpFile)

    def viewFile(self):
        options = ''
        gv_view_options = self.options.gvOptions
        abcTmpFile = tempfile.mktemp()
        psTmpFile = tempfile.mktemp()
        self.writeFile(self.getAbc(),abcTmpFile)

        
        self.abc2ps(abcTmpFile + ' -O ' + psTmpFile + '  ' + options)
        self.addTotalNbPages(psTmpFile)
        self.gv(gv_view_options + ' ' + psTmpFile)



    def isTune(self,string):
        if len(string) < 2 or string[:2] != 'X:':
            return False
        else:
            return True

    def tuneNumberKeep(self, tuneNumbers):
        split = tuneNumbers.split(',')
        tuneNumbers = []
        for i in split:
            tuneNumbers.append(int(i))
        abcsNew = []
        i = 0
        for abc in self.abcs:
            i = i + 1
            if i in tuneNumbers:
                abcsNew.append(abc)

        self.abcs = abcsNew
        
    def XtuneNumberKeep(self, XtuneNumbers):
        split = XtuneNumbers.split(',')
        XtuneNumbers = []
        for i in split:
            XtuneNumbers.append(int(i))
        abcsNew = []
        for abc in self.abcs:
            if int(abc.getX()) in XtuneNumbers:
                abcsNew.append(abc)
                
        self.abcs = abcsNew
        
    def getAbc(self):
        abc = []
        for tune in self.abcs:
            abc.append(tune.getAbc())
        return "\n\n".join(abc)


    def printAbc(self):
        print(self.getAbc())

    def listVoices(self):
        for abc in self.abcs:
            abc.listVoices()

    def listTitles(self):
        i = 1
        for abc in self.abcs:
            abc.listTitles(i)
            i += 1

    def keepVoices(self):
        for abc in self.abcs:
            abc.keepVoices(self.options.keepVoices)


    def explicitAccidentals(self):
        for abc in self.abcs:
            abc.explicitAccidentals()

    """
    def removeChords(self):
        for abc in self.abcs:
            abc.removeChords()

    def removeAnnotations(self):
        for abc in self.abcs:
            abc.removeAnnotations()
    """

    # lyrics
    def removeExtraLyrics(self):
        for abc in self.abcs:
            abc.removeExtraLyrics()

    def removeLyrics(self):
        for abc in self.abcs:
            abc.removeLyrics()

    def addTitle(self):
        for abc in self.abcs:
            abc.addTitle(self.options.addTitle)

    def extractLyrics(self):
        for abc in self.abcs:
            abc.extractLyrics()
        sys.exit()

    # end lyrics

    def removeChords(self):
        for abc in self.abcs:
            abc.removeChords()

    def removeAnnotations(self):
        for abc in self.abcs:
            abc.removeAnnotations()

    def whiteChordRoots(self):
        for abc in self.abcs:
            abc.whiteChordRootsTune()

    def danishChords(self):
        for abc in self.abcs:
            abc.danishChords()

    def replaceStrings(self):
        for abc in self.abcs:
            abc.replaceStrings(self.options.replaceStrings)
            
    def removeFingerings(self):
        for abc in self.abcs:
            abc.removeFingerings()

    def removeDecorations(self):
        for abc in self.abcs:
            abc.removeDecorations()

    def jazzChords(self):
        for abc in self.abcs:
            abc.jazzChords()

    def externalTranspose(self):
        for abc in self.abcs:
            abc.externalTranspose(self.options.externalTranspose)

    def transpose(self):
        for abc in self.abcs:
            abc.transpose(self.options.transpose)





class abctune:
    abc = ''

    voices = []
    
    root_splitter = re.compile('[a-gA-G]')

    key_aliases = {'': 'maj', 'none': ''}

    modes = {'ionian':0, 'ion': 0,
             'dorian':1, 'dor':1,
             'phrygian':2, 'phr':2,
             'lydian':3, 'lyd':3,
             'mixolydian':4, 'mix':4,
             'aeolian':5, 'aeo':5,
             'locrian':6, 'loc':6,
             'maj': 0,
             'm':5, 'minor':5, 'min': 5
             }

    white_keys = ['C','D','E','F','G','A','B']
    
    scales = {'': ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'],
              'm': ['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B']
              }
    scales['major'] = scales['']
    

    accidentals = {'C' : [],
                   'Db' : ['Db','Eb','Gb','Ab','Bb'],
                   'C#' : ['C#','D#','E#','F#','G#','A#','B#'],
                   'D' : ['F#','C#'],
                   'Eb' : ['Eb','Ab','Bb'],
                   'E' : ['F#','G#','C#','D#'],
                   'Fb' : ['Fb','Gb','Ab','Bbb','Cb','Db','Eb'],
                   'F' : ['Bb'],
                   'F#' : ['F#','G#','A#','C#','D#','E#'],
                   'Gb' : ['Gb','Ab','Bb','Cb','Db','Eb'],
                   'G' : ['F#'],
                   'Ab' : ['Ab','Bb','Db','Eb'],
                   'G#' : ['G#','A#','B#','C#','D#','E#','F##'],
                   'A' : ['C#','F#','G#'],
                   'Bbb' : ['Bbb','Cb','Db','Ebb','Fb','Gb','Ab'],
                   'Bb' : ['Bb','Eb'],
                   'B' : ['C#','D#','F#','G#','A#'],
                   'Cb' : ['Cb','Db','Eb','Fb','Gb','Ab','Bb'],
                   }


    
    def __init__(self, string):
        self.abc = string
        self.setVoices()



    def writeAbcToFile(self,file):
        self.writeFile(self.abc,file)
        
    def writeFile(self,string,file):
        f = open(file,'w')
        f.write(string)
        f.close()

    """
    def readFile(self, file):
        f = open(file,'r')
        filecontent = f.read()
        f.close()
        tunes = filecontent.split("\n\n")
        for tune in tunes:
            if  self.isTune(tune):
                self.abcs.append(abctune(tune))
    """


    def getX(self):
        X = -1
        for line in self.getLines():
            if line.strip()[:2] == 'X:':
                X = line.strip()[2:]
        return X

    def getAbc(self):
        return self.abc

    def abc2lines(self,string):
        return string.split("\n")

    def lines2abc(self,lines):
        return "\n".join(lines)

    def getLines(self):
        return self.abc2lines(self.abc)


    def parseMusicLine(self,line):
        parsedLine = []
        note = re.compile('[\^_]*[a-gA-G]')
        while len(line) > 0:
            node = None
            if note.match(line):
                note.match(line).group()
                node = {'type':'note', 'abc':note.match(line).group()}
                line = note.sub('',line,1)
            else:
                node = {'type':'misc', 'abc':line[0]}
                parsedLine.append(node)
                line = line[1:]
            if node:
                parsedLine.append(node)
        print(line)
        print(parsedLine)
        sys.exit()



    def generateMidi(self,midiFile):
        abcTmpFile = tempfile.mktemp()
        self.writeAbcToFile(abcTmpFile)
        os.system(external_programs['abc2midi'] + ' ' + abcTmpFile + ' -o ' + midiFile)
        os.remove(abcTmpFile)

    def generateXml(file,filename):
        remove_words(file)
        remove_extra_words(file)
        os.system(external_programs['abc2xml'] + ' -o ' + filename + ' ' + file)

    def playMidi(self):
        tmpfile = tempfile.mktemp();
        self.generateMidi(tmpfile)
        os.system(external_programs['midiplayer'] + ' ' + tmpfile)
        os.remove(tmpfile)

    def getLineType(self, line):
        line = line.strip()
        if len(line) > 1 and line[0].isupper() and line[1] == ':':
            return line[:2]
        elif line[:3] == '[V:':
            return '[V:'
        elif line == '':
            return 'blank'
        elif line[:2] == 'w:':
            return 'w:'
        elif line[:8] == '%%staves':
            return '%%staves'
        elif line.split(' ')[0] in ['%%begintext','%%endtext']:
            return 'pseudocomment'
        elif line[:1] == '%':
            return 'comment'
        else:
            return 'music'


    def removeExtraLyrics(self):
        lines = self.getLines()
        self.abc = ''
        for line in lines:
            if line[:2] != 'W:':
                self.abc += '\n' + line


    def addTitle(self,title):
        lines = self.getLines()
        self.abc = ''
        inside_title = False
        was_title = False
        for line in lines:
            if line[:2] == 'T:':
                inside_title = True
                was_title = True
            else:
                inside_title = False
            if (not inside_title) and was_title:
                self.abc += '\nT:' + title + '\n' + line
                was_title = False
            else:
                self.abc += '\n' + line

    def replaceStrings(self,replaceStrings):
        lines = self.getLines()
        replaceStrings = replaceStrings.split(',')
        while len(replaceStrings) > 1:
            i = 0
            for line in lines:
                lines[i] = line.replace(replaceStrings[0],replaceStrings[1])
                i = i + 1
            replaceStrings = replaceStrings[2:]
        self.abc = '\n'.join(lines)


    def explicitAccidentals(self):
        lines = self.getLines()
        modifiedLines = []
        pastHeader = False
        for line in lines:
            if self.getLineType(line) == 'K:':
                pastHeader = True
            elif pastHeader and self.getLineType(line) == 'music':
                parsedLine = self.parseMusicLine(line)
            modifiedLines.append(line)
        sys.exit()

    def removeLyrics(self):
        lines = self.getLines()
        self.abc = ''
        for line in lines:
            if line[:2] != 'w:':
                self.abc += '\n' + line

    def extractLyrics(self):
        slash_save = re.compile('-$')
        slash_replace = re.compile(' *- *')
        lyrics = ''
        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + '\n'
            i = i + 1
        file =''
        for line in lines:
            line = line.strip()
            if line[:2] == 'w:' or line[:2] == 'W:':
                current_line = line[2:]
                current_line = current_line.replace('*','')
                current_line = slash_save.sub('_____ATTEHACK_____',current_line)
                current_line = slash_replace.sub('',current_line)
                current_line = current_line.replace('_____ATTEHACK_____','-')
                current_line = current_line.strip()
                if current_line != '':
                    lyrics += current_line + "\n"
            elif line[:2] == 'P:' and lyrics != '':
                lyrics += '\n'
        print(lyrics[:-1])

    def stavesKeepVoices(self,line,voicesToKeep):
        cleanup1 = re.compile('\( *\)')
        cleanup2 = re.compile('\{ *\}')
        for voice in self.voices:
            if voice not in voicesToKeep:
                line = line.replace(voice,'')
        line = cleanup1.sub('',line)
        line = cleanup2.sub('',line)
        return line
        
    def keepVoices(self,voicesToKeep):
        voiceRE = re.compile('\[V\:([^\]]*)\]')
        voiceCleanAfterRE = re.compile('[\] ].*')
        voicesToKeep = voicesToKeep.split(',')
        lines = self.getLines()
        newLines = []
        past_header = False
        voiceName = ''
        removeThisLine = False
        protect = False
        for line in lines:
            split = voiceRE.split(line)
            if len(split) == 1:
                newLines.append(split[0])
            elif line.strip()[:3] == '[V:':
                line = split[1:]
                while len(line) > 0:
                    if len(line) > 2:
                        newLines.append('[V:' + line[0] + ']' + line[1] + '\\')
                    else:
                        newLines.append('[V:' + line[0] + ']' + line[1])
                    line = line[2:]
            else:
                line = split[1:]
                newLines.append(split[0] + '\\')
                while len(line) > 0:
                    if len(line) > 2:
                        newLines.append('[V:' + line[0] + ']' + line[1] + '\\')
                    else:
                        newLines.append('[V:' + line[0] + ']' + line[1])
                    line = line[2:]

        lines = newLines
        newLines = []
        for line in lines:
            lineType = self.getLineType(line)
            removeThisLine = False
            if lineType == 'K:':
                past_header = True
            elif lineType == '%%staves':
                line = self.stavesKeepVoices(line,voicesToKeep)
            elif past_header:
                if lineType == 'V:':
                    voice = line.replace('V:','').strip()
                    voice = voice.split(' ')
                    if len(voice) > 0:
                        voiceName = voice[0]
                elif lineType == '[V:':
                    voiceName = voiceCleanAfterRE.split(line.replace('[V:','').strip())[0]
                elif line.strip().split(' ')[0] in ['%%begintext']:
                    protect = True
                elif line.strip().split(' ')[0] in ['%%endtext']:
                    protect = True

                if voiceName not in voicesToKeep and voiceName != '' and lineType in ['music','V:','[V:','W:','w:'] and not protect:
                    removeThisLine = True
                else:
                    removeThisLine = False
            if not removeThisLine:
                newLines.append(line)
        self.abc = '\n'.join(newLines)
#        sys.exit()
        
        
    def setVoices(self):
        voiceRE = re.compile('\[V\:([^\]]*)\]')
        voices = []
        lines = self.getLines()
        for line in lines:
            if line.strip()[:2] == 'V:':
                voice = line.replace('V:','').strip()
                voice = voice.split(' ')
                if len(voice) > 0:
                    voiceName = voice[0]
                    if voiceName not in voices:
                        voices.append(voiceName)
            else:
                match = voiceRE.match(line)
                if match != None:
                    voiceName = match.group(1).split(' ')[0]
                    if voiceName not in voices:
                        voices.append(voiceName)
                
        self.voices = voices

        
    def listVoices(self):
        for voice in self.voices:
            print(voice)

    def listTitles(self, i):
        lines = self.abc.split('\n')
        for line in lines:
            line = line.strip()
            if line[:2] == 'T:':
                print str(i) + ':' + line[2:].strip()
                break

    def removeChordsFromLine(self,line):
        split = line.split('"')
        if len(split) > 1:
            line = ''
            i = 0
            for cell in split:
                if (i % 2) == 0:
                    line = line + cell
                elif len(cell) > 0 and cell[0] == '@':
                    line = line + '"' + cell + '"'
                i = i + 1
        return line

    def removeChords(self):
        lines = self.getLines()
        past_header = None
        p = re.compile('"[^@][^"]*"')
        lineNb = 0
        for line in lines:
            if line[:2] == 'K:':
                past_header = 1
            if past_header:
                #lines[lineNb] = p.sub('',line)
                lines[lineNb] = self.removeChordsFromLine(line)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

        

    def removeAnnotations(self):
        lines = self.getLines()
        past_header = None
        p = re.compile('"[@][^"]*"')
        lineNb = 0
        for line in lines:
            if line[:2] == 'K:':
                past_header = 1
            if past_header:
                lines[lineNb] = p.sub('',line)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

        

    def danishChords(self):
        lines  = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + '\n'
            i = i + 1

        header_done = False
        abc = ''
        for line in lines:
            strip = line.strip()
            if header_done:
                line = self.danishChordsLine(line)
            elif len(strip) == 0 or strip[0] == 'K':
                header_done = True
            abc += line

        self.abc = abc


    def danishChordsLine(self, abc):
        i = 0
        result = '';
        abc = abc.split('"')
        while i < len(abc):
            if i % 2 == 1:
                if abc[i][:1] == '@':
                    subst = abc[i]
                else:
                    #subst = string.replace(abc[i],'Bb','bbbbbbb')
                    subst = abc[i].replace('Bb','bbbbbbb')
                    #subst = string.replace(subst,'B','H')
                    subst = subst.replace('B','H')
                    #subst = string.replace(subst,'bbbbbbb','Bb')
                    subst = subst.replace('bbbbbbb','Bb')
                result += '"'  + subst + '"'
            else:
                result += abc[i]
            i = i + 1
        return result

    def removeFingerings(self):
        lines = self.getLines()
        past_header = None
        p = re.compile('![12345]!')
        lineNb = 0
        for line in lines:
            if line[:2] == 'K:':
                past_header = 1
            if past_header and line.strip()[:2] != 'w:' and line.strip()[:2] != 'W:':
                lines[lineNb] = p.sub('',line)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def removeDecorations(self):
        lines = self.getLines()
        past_header = None
        d1 = re.compile('![^!]*!')
        d2 = re.compile('\+[^\+]*\+')
        lineNb = 0
        for line in lines:
            if line[:2] == 'K:':
                past_header = 1
            if past_header and line.strip()[:2] != 'w:' and line.strip()[:2] != 'W:':
                lines[lineNb] = d2.sub('',d1.sub('',line))
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)

    def jazzChords(self):
        lines = self.getLines()
        file =''
        past_header = None
        lineNb = 0
        for line in lines:
            if line[:2] == 'K:':
                past_header = 1
            if past_header:
                split = line.split('"')
                for i in range(1, len(split), 2):
                    new = split[i]
                    new = new.replace('dim7','°')
                    new = new.replace('m7b5','ø')
                    new = new.replace('maj','Δ')
                    split[i] = new
                lines[lineNb] = '"'.join(split)
            lineNb = lineNb + 1
        self.abc = self.lines2abc(lines)


    def externalTranspose(self,halfsteps):
        tmpfile1 = tempfile.mktemp();
        tmpfile2 = tempfile.mktemp();
        self.writeAbcToFile(tmpfile1)
        os.system(external_programs['abc2abc'] + ' ' + tmpfile1 + ' -e -t ' + str(halfsteps) + ' > ' + tmpfile2)
        f = open(tmpfile2,'r')
        filecontent = f.read()
        f.close()
        os.remove(tmpfile1)
        os.remove(tmpfile2)
        self.abc = filecontent



    def transpose(self,transpose):
        transpose = int(transpose)
        key = 'none'
        result = []
        org_applied_accidentals = {}
        applied_accidentals = {}
        in_chord = False
        not_music = ['V','K','P','%','w','W','T']
        note = re.compile('^[_^=]*[a-gA-G][,\']*')
        chord_root = re.compile('^[A-G][b#]*')
        annotation = re.compile('^"[><_^@hijcklmnopqrstuvxxyzæøåHIJKLMNOPQRSTUVWXYZÆØÅ].*?"')
        white_key_root = re.compile('[a-gA-G]')
        k = re.compile('^K: *')
        hp_key = re.compile('^K: *H[pP]')
        tmp_save_key = re.compile('[^ ]*')
        ik = re.compile('^\[ *K: *')
        ik_end = re.compile('^.*?\]')
        deco = re.compile('^\!.*?\!')
        new_deco = re.compile('^\+.*?\+')
        sqr = re.compile('^\[[KMIV].*?\]')

        skip = False

        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + '\n'
            i = i + 1
        file =''

        # easy solution for K:Hp and K:HP suggested by Hudson Lacerda
        new_lines = []
        for line in lines:
            strip = line.strip()
            if hp_key.match(strip) != None:
                new_lines.append('%' + line)
                new_lines.append('K:Amix\n')
            else:
                new_lines.append(line)
        lines = new_lines

        voice = ''
        voice_entry_key = {}
        previous_key = ''
        header_done = False
        for line in lines:
            strip = line.strip()
            if len(strip) == 0:
                header_done = False
                key = 'none'
                result.append(line)
            elif strip[:11] == '%%begintext':
                skip = True
                result.append(line)
            elif strip[:9] == '%%endtext':
                skip = False
                result.append(line)
            elif skip:
                result.append(line)
            elif strip[0] == 'K' and not header_done:
                header_done = True
                key_res = k.match(strip)
                key = strip[key_res.span()[1]:]
                # this stips off everything after first space
                # FIXME; should handle modes and exp, UPDATE: it does handle modes
                # tmp_saved_key = tmp_save_key.match(key)
                # key = key[:tmp_saved_key.span()[1]]
                
                #print(key)
                #sys.exit()
                key_setup = self.setup_key(key,transpose)
                key = key_setup['key']
                new_key = key_setup['new_key']
                key_root = key_setup['key_root']
                key_type = key_setup['key_type']
                source_scale = key_setup['source_scale']
                target_scale = key_setup['target_scale']
                source_chord_scale = key_setup['source_chord_scale']
                target_chord_scale = key_setup['target_chord_scale']
                result.append('K:' + new_key + '\n')
            elif strip[0] == 'P':
                org_applied_accidentals = {}
                applied_accidentals = {}
                #voice_entry_key = {}
                result.append(line)
            elif strip[0] == 'V' and header_done:
                voice = re.sub(' .*', '', strip[2:].strip())
                org_applied_accidentals = {}
                applied_accidentals = {}
                if voice in voice_entry_key.keys() and voice_entry_key[voice] != '':
                    key = voice_entry_key[voice]
                    key_setup = self.setup_key(key,transpose)
                    key = key_setup['key']
                    new_key = key_setup['new_key']
                    key_root = key_setup['key_root']
                    key_type = key_setup['key_type']
                    source_scale = key_setup['source_scale']
                    target_scale = key_setup['target_scale']
                    source_chord_scale = key_setup['source_chord_scale']
                    target_chord_scale = key_setup['target_chord_scale']
                else:
                    voice_entry_key[voice] = key
                result.append(line)
            elif not header_done or strip[0] in not_music:
                result.append(line)
            else:
                while len(line) > 0:
                    next = None
                    note_res = note.match(line)
                    inkey_res = ik.match(line)
                    deco_res = deco.match(line)
                    new_deco_res = new_deco.match(line)
                    sqr_res = sqr.match(line)
                    annotation_res = annotation.match(line)
                    if line[0] == '|':
                        org_applied_accidentals = {}
                        applied_accidentals = {}
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif annotation_res != None and not in_chord:
                        annotation_length = annotation_res.span()[1]
                        next = line[:annotation_length]
                        result.append(next)
                        line = line[annotation_length:]
                    elif line[0] == '"':
                        in_chord = not in_chord
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif in_chord:
                        chord_res = chord_root.match(line)
                        if chord_res != None:
                            chord_length = chord_res.span()[1]
                            next = line[:chord_length]
                            next = self.transpose_in_scales(next,source_chord_scale,target_chord_scale)
                            result.append(next)
                            line = line[chord_length:]
                        else:
                            next = line[0]
                            result.append(next)
                            line = line[1:]

                    elif inkey_res != None:
                        k_start = line[:inkey_res.span()[1]]
                        result.append(k_start)
                        line = line[inkey_res.span()[1]:]
                        inkey_end_res = ik_end.match(line)
                        if voice in voice_entry_key.keys() and voice_entry_key[voice] == '':
                            voice_entry_key[voice] = key
                        key = line[:inkey_end_res.span()[1] - 1]
                        voice_entry_key[voice] = key
                        
                        # this stips off everything after first space
                        # FIXME; should handle modes and exp
                        tmp_saved_key = tmp_save_key.match(key)
                        key = key[:tmp_saved_key.span()[1]]

                        key_setup = self.setup_key(key,transpose)
                        key = key_setup['key']
                        new_key = key_setup['new_key']
                        key_root = key_setup['key_root']
                        key_type = key_setup['key_type']
                        source_scale = key_setup['source_scale']
                        target_scale = key_setup['target_scale']
                        source_chord_scale = key_setup['source_chord_scale']
                        target_chord_scale = key_setup['target_chord_scale']

                        result.append(new_key)

                        line = line[inkey_end_res.span()[1] - 1:]
                    elif deco_res != None:
                        next = line[:deco_res.span()[1]]
                        result.append(next)
                        line = line[deco_res.span()[1]:]
                    elif new_deco_res != None:
                        next = line[:new_deco_res.span()[1]]
                        result.append(next)
                        line = line[new_deco_res.span()[1]:]
                    elif sqr_res != None:
                        next = line[:sqr_res.span()[1]]
                        result.append(next)
                        line = line[sqr_res.span()[1]:]
                    elif note_res != None:
                        note_length = note_res.span()[1]
                        next = line[:note_length]
                        next = self.normalize_octave(next)
                        if key_type == 'none':
                            next_res = white_key_root.search(next)
                            next_root = next[next_res.span()[0]:]
                            next_accidental = next[:next_res.span()[0]]
                            if next_accidental != '':
                                org_applied_accidentals[next_root] = next_accidental
                            else:
                                if next_root not in org_applied_accidentals.keys():
                                    org_applied_accidentals[next_root] = '='
                            next = org_applied_accidentals[next_root] + next_root
                            next = self.transpose_in_scales(next,source_scale,target_scale)
                            next_res = white_key_root.search(next)
                            next_root = next[next_res.span()[0]:]
                            next_accidental = next[:next_res.span()[0]]
                            if next_accidental == '=' and next_root not in applied_accidentals.keys():
                                applied_accidentals[next_root] = '='
                            
                            if next_root in applied_accidentals.keys() and applied_accidentals[next_root] == next_accidental:
                                next = next_root
                            applied_accidentals[next_root] = next_accidental
                        else:
                            next = self.transpose_in_scales(next,source_scale,target_scale)
                        result.append(next)
                        line = line[note_length:]
                    else:
                        next = line[0]
                        result.append(next)
                        line = line[1:]
        
        result = ''.join(result)
        
        self.abc = result.strip('\n')


    def setup_key(self,key,transpose):
        # remove %comment after key
        comment = re.compile('%.*')
        key = comment.sub('',key).strip()

        root = re.compile('^[A-G][b#]?')
        none = re.compile('none')
        #strip = string.lower(key).strip()
        strip = key.lower().strip()
        if strip in ['none','']:
            key = 'Cnone' # Hmmm, what's up with this, changed to...
            #key = 'C'
        else:
            # look for alternate ways to write keys
            root_res = root.match(key)
            #print(root_res)
            #sys.exit()
            key_root = key[:root_res.span()[1]]
            key_type = key[root_res.span()[1]:]
            #key_type = string.lower(key_type).strip()
            key_type = key_type.lower().strip()
            if key_type != '' and key_type in self.modes:
                for real_key_type in self.key_aliases:
                    if key_type == self.key_aliases[real_key_type]:
                        key_type = real_key_type
                        key = key_root + key_type
                        break
            else:
                if key_type == '':
                    key = key_root
                else:
                    exit
            
            # make F# -> Gb
            key = self.non_enharmonic_key(key)
        # recalculate key_root, in case it was changed in non_enharmonic_key()
        root_res = root.match(key)
        key_root = key[:root_res.span()[1]]
        key_type = key[root_res.span()[1]:]

        result = {}

        source_scale = self.find_scale(key)
        source_chord_scale = self.find_chord_scale(key)
        new_key = self.find_key(key,transpose)
        target_scale = self.find_scale(new_key)
        scale_offset = self.get_midi_key(source_scale[0]) - self.get_midi_key(target_scale[0])
        if (scale_offset + transpose) % 12 == 0:
            while scale_offset + transpose != 0:
                if scale_offset + transpose > 0:
                    target_scale = target_scale[42:]
                else:
                    source_scale = source_scale[42:]

                scale_offset = self.get_midi_key(source_scale[0]) - self.get_midi_key(target_scale[0])
        else:
            pass
            

        target_chord_scale = self.find_chord_scale(new_key)


        none_res = none.search(new_key)
        if none_res != None:
            new_key = 'none'

        result['key'] = key
        result['new_key'] = new_key
        result['source_scale'] = source_scale
        result['target_scale'] = target_scale
        result['source_chord_scale'] = source_chord_scale
        result['target_chord_scale'] = target_chord_scale
        result['key_root'] = key_root
        result['key_type'] = key_type

        return result


    def non_enharmonic_key(self,key):
        if key == 'F#':
            result = 'Gb'
        elif key == 'D#m':
            result = 'Ebm'
        elif key == 'D#minor':
            result = 'Ebminor'
        else:
            result = key

        return result
    



    def find_scale(self,key):
        none = False
        root = re.compile('^[A-G][b#]?')
        root_res = root.match(key)
        scale_root = key[:root_res.span()[1]]
        scale_type = key[root_res.span()[1]:]
        actual_scale = self.map_key_to_accidentals(key)
        actual_scale = actual_scale.split(' ')[0]
        

        if scale_type == 'major':
            key = scale_root
        if scale_type == 'none':
            key = scale_root
            none = True
        result = []
        octaves = [',,,,,,',',,,,,',',,,,',',,,',',,',',','','lower',"'","''","'''","''''","'''''"]
        offset = self.white_keys.index(key[0])
        i = offset
        lower = False
        for modifier in octaves:
            if modifier == 'lower':
                lower = True
                modifier = ''
            while i < 7:
                white_key = self.white_keys[i]
                real_key = self.apply_accidentals(white_key,self.accidentals[actual_scale])
                if lower:
                    #white_key = string.lower(white_key)
                    white_key = white_key.lower()
                white_key = white_key + modifier
                if len(real_key) == 1:
                    if none:
                        result.append('__' + white_key)
                        result.append('_' + white_key)
                        result.append('=' + white_key)
                        result.append('^' + white_key)
                        result.append('^^' + white_key)
                        result.append('')
                    else:
                        result.append('__' + white_key)
                        result.append('_' + white_key)
                        result.append(white_key)
                        result.append('=' + white_key)
                        result.append('^' + white_key)
                        result.append('^^' + white_key)
                elif real_key[1] == 'b':
                    if none:
                        result.append('___' + white_key)
                        result.append('__' + white_key)
                        result.append('_' + white_key)
                        result.append('=' + white_key)
                        result.append('^' + white_key)
                        result.append('')
                    else:
                        result.append('___' + white_key)
                        result.append('__' + white_key)
                        result.append(white_key)
                        result.append('_' + white_key)
                        result.append('=' + white_key)
                        result.append('^' + white_key)
                elif real_key[1] == '#':
                    if none:
                        result.append('_' + white_key)
                        result.append('=' + white_key)
                        result.append('^' + white_key)
                        result.append('^^' + white_key)
                        result.append('^^^' + white_key)
                        result.append('')
                    else:
                        result.append('_' + white_key)
                        result.append('=' + white_key)
                        result.append(white_key)
                        result.append('^' + white_key)
                        result.append('^^' + white_key)
                        result.append('^^^' + white_key)
                else:
                    pass
                i = i + 1
            i = 0
        return result



    def map_key_to_accidentals(self,key):
        root = re.compile('^[A-G][b#]?')
        root_res = root.match(key)
        scale_root = key[:root_res.span()[1]]
        scale_type = key[root_res.span()[1]:]
        #scale_type = string.lower(scale_type).strip()
        scale_type = scale_type.lower().strip()

        if scale_type == 'none':
            return scale_root

        degree = None
        for i in self.modes:
            if scale_type == i:
                degree = self.modes[i]
                break

        if degree == None:
            return key

        for potential_key in self.accidentals:
            potential_scale_root = potential_key[0]
            start_on = self.white_keys.index(potential_scale_root)
            if self.apply_accidentals(self.white_keys[(degree + start_on) % 7],self.accidentals[potential_key]) == scale_root:
                return potential_key

        return key




    def apply_accidentals(self,white_key,accidentals):
        result = white_key
        for i in accidentals:
            if white_key == i[0]:
                result = i
                break
        return result



    def find_chord_scale(self,key):
        none = False
        root = re.compile('^[A-G][b#]?')
        root_res = root.match(key)
        scale_root = key[:root_res.span()[1]]
        scale_type = key[root_res.span()[1]:]
        if scale_type == 'major':
            key = scale_root
        if scale_type == 'none':
            key = scale_root
            none = True

        result = []
        offset = self.white_keys.index(key[0])
        degree = offset
        for i in range(7):
            white_key = self.white_keys[degree % 7]
            actual_scale = self.map_key_to_accidentals(key)
            actual_scale = actual_scale.split(' ')[0]
            real_key = self.apply_accidentals(white_key,self.accidentals[actual_scale])
            if len(real_key) == 1:
                result.append(white_key + 'bb')
                result.append(white_key + 'b')
                result.append(white_key)
                result.append(white_key + '#')
                result.append(white_key + '##')
            elif real_key[1] == 'b':
                result.append(white_key + 'bbb')
                result.append(white_key + 'bb')
                result.append(white_key + 'b')
                result.append(white_key)
                result.append(white_key + '#')
            elif real_key[1] == '#':
                result.append(white_key + 'b')
                result.append(white_key)
                result.append(white_key + '#')
                result.append(white_key + '##')
                result.append(white_key + '###')
            else:
                pass
            degree = degree + 1
        return result


    def find_key(self,source_key,transpose):
        none = False
        root = re.compile('^[A-G][b#]?')
        root_res = root.match(source_key)
        scale_root = source_key[:root_res.span()[1]]
        scale_type = source_key[root_res.span()[1]:]
        if scale_type == 'none':
            none = True
            scale_type = ''
        new_root = self.transpose_root(scale_root,scale_type,transpose)
        new_key = new_root + scale_type
        if none:
            new_key = new_key + 'none'
        return new_key


    def transpose_root(self,scale_root,scale_type,transpose):
        #scale_type = string.lower(scale_type).strip()
        scale_type = scale_type.lower().strip()
        if scale_type not in self.modes.keys() and scale_type not in self.scales.keys():
            scale_type = 'major'
            
        if scale_type in self.modes.keys():
            actual_scale = self.map_key_to_accidentals(scale_root + scale_type)
            new_actual_scale = self.find_key(actual_scale,transpose)
            new_root = self.white_keys[(self.white_keys.index(new_actual_scale[0]) + self.modes[scale_type]) % 7]
            new_root = self.apply_accidentals(new_root,self.accidentals[new_actual_scale])

        else:
            #herher
            #print(scale_type)
            #print(self.scales)
            #print(self.scales[scale_type])
            #sys.exit()
            index = self.scales[scale_type].index(scale_root)
            new_root = self.scales[scale_type][(index + transpose) % 12]
        return new_root



    def get_midi_key(self,note):
        # doesn't work with any abc note, since C could mean C# in A major, etc
        # but since it's only used for the lowest note in a scale where explicit
        # flats are always present, that's ok...
        midi_notes = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
                      'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83}

        root = re.compile('[a-gA-G]')
        root_res = root.search(note)
        root =  note[root_res.span()[0]:root_res.span()[1]]
        midi_key = midi_notes[root]

        #midi_key = midi_key - string.count(note,'_')
        #midi_key = midi_key - string.count(note,'^')
        #midi_key = midi_key - (string.count(note,',') * 12)
        #midi_key = midi_key + (string.count(note,"'") * 12)

        midi_key = midi_key - note.count('_')
        midi_key = midi_key - note.count('^')
        midi_key = midi_key - (note.count(',') * 12)
        midi_key = midi_key + (note.count("'") * 12)

        return midi_key


    def transpose_in_scales(self,note,source_scale,target_scale):
        index = source_scale.index(note)
        result = target_scale[index]
        return result


    def normalize_octave(self,note):
        root_res = self.root_splitter.search(note)
        span = root_res.span()
        accidentals = note[:span[0]]
        root = note[span[0]:span[1]]
        octave = note[span[1]:]

        octave_count = octave.count("'") - octave.count(',')
        if octave_count == 0:
            pass
        elif octave_count < 0 and root.islower():
            root = string.upper(root)
            octave_count = octave_count + 1
        elif octave_count > 0 and root.isupper():
            root = string.lower(root)
            octave_count = octave_count - 1

        if octave_count == 0:
            octave = ''
        elif octave_count > 0:
            octave = "'" * octave_count
        else:
            octave = ',' * (- octave_count)

        return accidentals + root + octave


    def whiteChordRoot(self,root):
        result = root
        if root == 'Cbbb':
            result = 'A'
        elif root == 'Cbb':
            result = 'Bb'
        elif root == 'Cb':
            result = 'B'
        elif root == 'C##':
            result = 'D'
            
        elif root == 'Dbb':
            result = 'C'
        elif root == 'D##':
            result = 'E'
            
        elif root == 'Ebb':
            result = 'D'
        elif root == 'E#':
            result = 'F'
        elif root == 'E##':
            result = 'F#'
        elif root == 'E###':
            result = 'G'
            
        elif root == 'Fbbb':
            result = 'D'
        elif root == 'Fbb':
            result = 'Eb'
        elif root == 'Fb':
            result = 'E'
        elif root == 'F##':
            result = 'G'
            
        elif root == 'Gbb':
            result = 'F'
        elif root == 'G##':
            result = 'A'
            
        elif root == 'Abb':
            result = 'G'
        elif root == 'A##':
            result = 'B'
            
            
        elif root == 'Bbb':
            result = 'A'
        elif root == 'B#':
            result = 'C'
        elif root == 'B##':
            result = 'C#'
        elif root == 'B###':
            result = 'D'
        return result





    def whiteChordRootsTune(self):
        result = []
        in_chord = False
        not_music = ['V','K','P','%','w','W']
        chord = re.compile('^[A-G][b#]*')
        annotation = re.compile('^"[><_^@].*?"')

        lines = self.getLines()
        i = 0
        for line in lines:
            lines[i] = line + '\n'
            i = i + 1

        voice_entry_key = ''
        previous_key = ''
        header_done = False
        for line in lines:
            strip = line.strip()
            if len(strip) == 0:
                header_done = False
                result.append(line)
            elif strip[0] == 'K' and not header_done:
                header_done = True
                result.append(line)
            elif not header_done or strip[0] in not_music:
                result.append(line)
            else:
                while len(line) > 0:
                    next = None
                    annotation_res = annotation.match(line)
                    if line[0] == '|':
                        org_applied_accidentals = {}
                        applied_accidentals = {}
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif annotation_res != None and not in_chord:
                        annotation_length = annotation_res.span()[1]
                        next = line[:annotation_length]
                        result.append(next)
                        line = line[annotation_length:]
                    elif line[0] == '"':
                        in_chord = not in_chord
                        next = line[0]
                        result.append(next)
                        line = line[1:]
                    elif in_chord:
                        chord_res = chord.match(line)
                        if chord_res != None:
                            chord_length = chord_res.span()[1]
                            next = line[:chord_length]
                            next = self.whiteChordRoot(next)
                            result.append(next)
                            line = line[chord_length:]
                        else:
                            next = line[0]
                            result.append(next)
                            line = line[1:]
                    else:
                        next = line[0]
                        result.append(next)
                        line = line[1:]
        result = ''.join(result)

        self.abc = result





abc = abctool()
abc.parseOptions()
abc.process()




