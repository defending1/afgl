\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mathreport}[2023/10/27 Custom Typographic Math Report Class]

% Pass unknown options to the base article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt, letterpaper]{article}

% =========================================================
%  1. TYPOGRAPHY & FONTS (The Foundation)
% =========================================================
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}

% Palatino (Text and Math) with Old Style Figures
\RequirePackage[osf,sc]{mathpazo} 
\RequirePackage{tgpagella} 

% Open up the leading (line spacing) for readability
\linespread{1.15}

% Microtype: The secret sauce for perfect justification
\RequirePackage[activate={true,nocompatibility},final,tracking=true,kerning=true,spacing=true]{microtype}
\SetTracking{encoding={*}, shape=sc}{40} % Spacing for Small Caps

% =========================================================
%  2. PAGE GEOMETRY (The Measure)
% =========================================================
% We enforce wide margins to keep line length ~66 characters
\RequirePackage[top=1.5in, bottom=1.5in, left=1.8in, right=1.8in]{geometry}

% =========================================================
%  3. MATHEMATICS
% =========================================================
\RequirePackage{amsmath, amssymb, amsthm, mathtools, bm}

% Custom Theorem Style: Italic body, Small Caps header
\newtheoremstyle{eleganttheorem}
  {12pt}{12pt}{\itshape}{}{\scshape}{.}{0.5em}{}

\theoremstyle{eleganttheorem}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]

% =========================================================
%  4. SECTION HEADINGS
% =========================================================
\RequirePackage{titlesec}

% Section: Centered, Large, Small Caps
\titleformat{\section}[block]{\large\scshape\centering}{\thesection.}{1em}{}
\titlespacing*{\section}{0pt}{2em}{1.5em}

% Subsection: Left aligned, Italic
\titleformat{\subsection}[block]{\normalsize\itshape}{\thesubsection}{1em}{}
\titlespacing*{\subsection}{0pt}{1.5em}{0.5em}

% =========================================================
%  5. UTILITIES
% =========================================================
\RequirePackage{booktabs} % Good tables
\RequirePackage{caption}  % Caption styling
\captionsetup{font={small,it}, labelfont={sc}, labelsep=period}
\RequirePackage[dvipsnames]{xcolor}

% Hyperlinks
\RequirePackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=Maroon,
    citecolor=Maroon,
    urlcolor=MidnightBlue
}
\RequirePackage{cleveref}

% Bibliography
\RequirePackage[style=authoryear, backend=biber]{biblatex}
