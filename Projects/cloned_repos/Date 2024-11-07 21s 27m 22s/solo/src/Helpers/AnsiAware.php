<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Helpers;

class AnsiAware
{
    public static function wordwrap($string, $width = 75, $break = PHP_EOL, $cut = false): string
    {
        $ansiEscapeSequence = '/(\x1b\[[0-9;]*[mGK])/';
        $wordsPattern = '/(\S+\s+)/';

        // Split the string into an array of printable characters and ANSI codes
        $parts = preg_split($ansiEscapeSequence, $string, -1, PREG_SPLIT_DELIM_CAPTURE | PREG_SPLIT_NO_EMPTY);

        $lines = [];
        $currentLine = '';
        $currentLength = 0;
        $openAnsiCodes = '';

        foreach ($parts as $part) {
            if (preg_match($ansiEscapeSequence, $part)) {
                // ANSI code, append without affecting length
                $currentLine .= $part;

                // Update the openAnsiCodes
                if (str_contains($part, 'm')) { // SGR (Select Graphic Rendition) codes
                    if ($part == "\e[0m") {
                        // Reset code, clear openAnsiCodes
                        $openAnsiCodes = '';
                    } else {
                        // Add to openAnsiCodes
                        $openAnsiCodes .= $part;
                    }
                }

                continue;
            }

            // Split the part into words or characters based on $cut
            $wordsOrChars = $cut
                // Cut the string at exact length
                ? mb_str_split($part)
                // Split the part into words
                : preg_split($wordsPattern, $part, -1, PREG_SPLIT_DELIM_CAPTURE | PREG_SPLIT_NO_EMPTY);

            foreach ($wordsOrChars as $wordOrChar) {
                $length = mb_strlen($wordOrChar);

                if ($currentLength + $length > $width) {
                    // Exceeds the width, wrap to the next line
                    if ($currentLine !== '') {
                        // Close any open ANSI codes
                        if ($openAnsiCodes !== '') {
                            $currentLine .= "\e[0m";
                        }
                        $lines[] = $currentLine;
                    }
                    // Start new line with open ANSI codes
                    $currentLine = $openAnsiCodes . $wordOrChar;
                    $currentLength = $length;
                } else {
                    // Append the character to the current line
                    $currentLine .= $wordOrChar;
                    $currentLength += $length;
                }
            }
        }

        // Append any remaining text
        if ($currentLine !== '') {
            // Close any open ANSI codes
            if ($openAnsiCodes !== '') {
                $currentLine .= "\e[0m";
            }

            $lines[] = $currentLine;
        }

        return implode($break, $lines);
    }
}
