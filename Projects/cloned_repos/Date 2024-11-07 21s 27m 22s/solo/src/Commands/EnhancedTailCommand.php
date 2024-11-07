<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Commands;

use AaronFrancis\Solo\Facades\Solo;
use Illuminate\Support\Collection;
use Illuminate\Support\Str;

class EnhancedTailCommand extends Command
{
    public function wrapAndFormat($line): string|array
    {
        $theme = Solo::makeTheme();

        // A single trailing line that closes the JSON exception object.
        if (trim($line) === '"}') {
            return '';
        }

        if (str_contains($line, '{"exception":"[object] ')) {
            return $this->formatInitialException($line);
        }

        if (str_contains($line, '[stacktrace]')) {
            return '   ' . $theme->dim($line);
        }

        if (!Str::isMatch('/#[0-9]+ /', $line)) {
            return $this->wrapLine($line);
        }

        // Make the line shorter by removing the base path. Helps prevent wrapping.
        $line = str_replace(base_path(), '', $line);

        // Replace all vendor frame with a simple placeholder.
        // @TODO make this configurable? Or toggleable?
        if (str_contains($line, '/vendor/') && !Str::isMatch("/BoundMethod\.php\([0-9]+\): App/", $line)) {
            return $theme->dim('   [Vendor frames]');
        }

        return array_map(function ($line) {
            return (Str::isMatch('/#[0-9]+ /', $line) ? str_repeat(' ', 3) : str_repeat(' ', 7)) . $line;
        }, $this->wrapLine($line, -7));
    }

    public function formatInitialException($line): array
    {
        $lines = explode('{"exception":"[object] ', $line);

        // Wrap first and then apply formatting, so that we don't have to
        // muck around with ANSI codes when trying to measure width.
        $message = collect($lines[0])
            ->flatMap($this->wrapLine(...))
            ->map(fn($line) => Solo::makeTheme()->red($line));

        $exception = collect($lines[1])
            // 3 for the 3 spaces we prepend.
            ->flatMap(fn($line) => $this->wrapLine($line, -3))
            ->map(fn($line) => '   ' . Solo::makeTheme()->exception($line));

        return [
            ...$message->toArray(), ...$exception->toArray()
        ];
    }

    protected function modifyWrappedLines(Collection $lines): Collection
    {
        $hasVendorFrame = false;

        // After all the lines have been wrapped, we look through them
        // to collapse consecutive vendor frames into a single line.
        return $lines->filter(function ($line) use (&$hasVendorFrame) {
            $isVendorFrame = str_contains($line, '[Vendor frames]');

            if ($isVendorFrame) {
                // Skip the line if a vendor frame has already been added.
                if ($hasVendorFrame) {
                    return false;
                }
                // Otherwise, mark that a vendor frame has been added.
                $hasVendorFrame = true;
            } else {
                // Reset the flag if the current line is not a vendor frame.
                $hasVendorFrame = false;
            }

            return true;
        });
    }
}
