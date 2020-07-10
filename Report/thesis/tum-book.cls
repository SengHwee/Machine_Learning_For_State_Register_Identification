%%
% TUM Corporate Design Book
% Michael Riesch (michael.riesch@tum.de)
% Based on the templates from https://www.tum.de/cd
%
% Template for scientific theses, lecture notes, ...
%
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tum-book}[2018/10/08 TUM Corporate Design Book v0.3]

% process options
\DeclareOption{a4paper}{%
  \PassOptionsToPackage{a4paper}{tum-base-text}
  \PassOptionsToClass{a4paper,fontsize=11pt}{scrbook}
}
\DeclareOption{a5paper}{%
  \PassOptionsToPackage{a5paper}{tum-base-text}
  \PassOptionsToClass{a5paper,fontsize=10pt}{scrbook}
}
\DeclareOption{twoside}{\PassOptionsToClass{twoside}{scrbook}}
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{scrbook}}
\DeclareOption{times}{\PassOptionsToPackage{times}{tum-base}}
\DeclareOption{german}{%
  \PassOptionsToPackage{german}{tum-base}
  \PassOptionsToPackage{german}{nomencl}
}
\DeclareOption{english}{%
  \PassOptionsToPackage{english}{tum-base}
  \PassOptionsToPackage{english}{nomencl}
}
\ExecuteOptions{twoside,a4paper}
\ProcessOptions\relax

% load base class
\LoadClass[pagesize,bibtotoc,idxtotoc,numbers=noenddot]{scrbook}

% load TUM base packages
\RequirePackage{tum-base}
\RequirePackage{tum-base-text}

\newcommand{\thepublisher}{~}
\newcommand{\publisher}[1]{\renewcommand{\thepublisher}{#1}}
\newcommand{\thetitledescription}{~}
\newcommand{\titledescription}[1]{\renewcommand{\thetitledescription}{#1}}
\newcommand{\thefineprint}{~}
\newcommand{\fineprint}[1]{\renewcommand{\thefineprint}{#1}}

% custom titlepage
\AtBeginDocument{
  \begin{titlepage}
    \thispagestyle{titlepage}
    \ifavier
    % title page a4
    \fontfamily{phv}\selectfont
    { \flushleft\Huge\bfseries\@title\par }
    { \flushleft\LARGE\@subtitle\par }
    \vspace{1cm}
    { \flushleft\Large\textcolor{TUMBlau}{\textbf{\@author}}\par
      \thepublisher\par }
    \vspace{1cm}
    { \flushleft\Large\thetitledescription\par}
    \vspace{1cm}
    { \flushleft\large\thefineprint\par }

    \begin{textblock*}{11cm}[1,1](\paperwidth-2\tumlogoheight,
      \paperheight-2\tumlogoheight)
      \raggedleft
      \includegraphics[width=11cm]{tum-resources/images/TUM_Uhrenturm.png}\par
    \end{textblock*}

    \else
    % title page a5
    \fontfamily{phv}\selectfont
    \vspace{12pt}
    { \flushleft\huge\bfseries\@title\par }
    { \flushleft\Large\@subtitle\par }
    \vspace{12pt}
    { \flushleft\Large\textcolor{TUMBlau}{\textbf{\@author}}\par
      \thepublisher\par }
    { \flushleft\large\thetitledescription\par }
    { \flushleft\normalsize\thefineprint\par }
    \vspace{12pt}

    \begin{textblock*}{8cm}[1,1](\paperwidth-2\tumlogoheight, \paperheight)
      \raggedleft
      \includegraphics[width=8cm]{tum-resources/images/TUM_Uhrenturm.png}\par
    \end{textblock*}

    \fi

    \vfill
  \end{titlepage}
}

% custom header/footer
\RequirePackage{fancyhdr}
\fancypagestyle{plain}{
  % clear everything
  \fancyhead{}
  \fancyfoot{}

  % page numbers
  \fancyfoot[LE,RO]{\fontfamily{phv}\selectfont\textbf{\thepage}}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

\fancypagestyle{titlepage}{%
  % clear everything
  \fancyhead{}
  \fancyfoot{}

  % threeliner
  \fancyhead[L]{%
    \vbox to\tumlogoheight{%
      \fontfamily{phv}\fontsize{9}{11}\selectfont\textcolor{TUMBlau}{%
        \thegroup\\
        \thedepartment\\
        \theuniversity
      }
    }
  }

  % logo
  \fancyhead[R]{%
    \vbox to\tumlogoheight{%
      \includegraphics[height=\tumlogoheight]%
                      {tum-resources/images/Universitaet_Logo_RGB.pdf}
    }
  }
}

\pagestyle{plain}

\newcommand{\printthesisdisclaimer}[2]{%
  \pagestyle{empty}
  \cleardoublepage
  \null\par\vfill
  {
    \fontfamily{phv}\selectfont
    \noindent
    I hereby declare that this thesis is entirely the result of my own work
    except where otherwise indicated. I have only used the resources given in
    the list of references.\par
    \vspace{15mm}
    \noindent
    #1 \hspace{5cm} #2\par
  }
  \newpage
  \pagestyle{plain}
}

\newcommand{\printdedication}[1]{%
  \thispagestyle{empty}
  \phantom{u}
  \phantom{1}\vspace{6cm}
  \begin{center}
    \fontfamily{phv}\selectfont
    #1
  \end{center}
  \newpage
  \pagestyle{plain}
}

\newcommand{\printphdthesistitle}[5]{%
  \pagestyle{empty}
  \cleardoublepage
  \pagestyle{titlepage}
  \begin{otherlanguage}{ngerman}
    { \flushleft\fontfamily{phv}\selectfont\large\thedepartment\par
      \thegroup\par
    }
    \vspace{12pt}
    { \flushleft\fontfamily{phv}\selectfont\huge\bfseries\@title\par }
    { \flushleft\flushleft\fontfamily{phv}\Large
      \textcolor{TUMBlau}{\textbf{\@author}}\par}
    { \flushleft\fontfamily{phv}\selectfont\normalsize
      Vollständiger Abdruck der von der \thedepartment{} der
      \theuniversity{} zur Erlangung des akademischen Grades eines\\
      \textbf{Doktor-Ingenieurs (Dr.-Ing.)}\\
      genehmigten Dissertation.\par
    }
    { \flushleft\fontfamily{phv}\selectfont\normalsize
      Vorsitzende(r): #3\par
      Prüfer der Dissertation:
      \begin{enumerate}[nolistsep]
      \item #4
      \item #5
      \end{enumerate}
    }
    \vspace{12pt}
    { \flushleft\fontfamily{phv}\selectfont\normalsize
      Die Dissertation wurde am #1 bei der \theuniversity{} eingereicht und
      durch die \thedepartment{} am #2 angenommen.\par}
  \end{otherlanguage}
  \newpage
  \pagestyle{plain}
}
