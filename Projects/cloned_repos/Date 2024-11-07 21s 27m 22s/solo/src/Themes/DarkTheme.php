<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Themes;

class DarkTheme extends LightTheme
{
    /*
    |--------------------------------------------------------------------------
    | Tabs
    |--------------------------------------------------------------------------
    */
    public function tabFocused(string $text): string
    {
        return $this->bgWhite($this->black($text));
    }

    public function tabBlurred(string $text): string
    {
        return $this->dim($text);
    }

    public function tabStopped(string $text): string
    {
        return ' ' . $this->strikethrough(trim($text)) . ' ';
    }

    /*
    |--------------------------------------------------------------------------
    | Logs
    |--------------------------------------------------------------------------
    */
    public function logsPaused(string $text): string
    {
        return $this->yellow($text);
    }

    /*
    |--------------------------------------------------------------------------
    | Process
    |--------------------------------------------------------------------------
    */
    public function processStopped(string $text): string
    {
        return $this->red($text);
    }
}
