<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Tests\Unit;

use AaronFrancis\Solo\Helpers\AnsiAware;
use Laravel\Prompts\Concerns\Colors;
use PHPUnit\Framework\Attributes\Test;

class AnsiAwareTest extends Base
{
    use Colors;

    #[Test]
    public function it_wraps_a_basic_line(): void
    {
        $line = str_repeat('a', 10);
        $width = 5;

        $wrapped = AnsiAware::wordwrap(string: $line, width: $width, cut: true);

        $this->assertEquals(
            "aaaaa\naaaaa",
            $wrapped
        );
    }

    #[Test]
    public function it_wraps_an_ansi_line(): void
    {
        $line = $this->bgRed($this->green(str_repeat('a', 10)));
        $width = 5;

        $wrapped = AnsiAware::wordwrap(string: $line, width: $width, cut: true);

        $this->assertEquals(
            "\e[41m\e[32maaaaa\e[0m\n\e[41m\e[32maaaaa\e[39m\e[49m\e[0m",
            $wrapped
        );
    }
}
