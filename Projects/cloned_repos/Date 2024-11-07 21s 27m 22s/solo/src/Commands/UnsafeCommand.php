<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Commands;

class UnsafeCommand extends Command
{
    public function logCaller($caller)
    {
        $message = <<<EOT
Cannot start potentially unsafe process added from [$caller]. 

To allow commands registered from this class to run, please add the following to either your AppServiceProvider or SoloServiceProvider: 

Solo::allowCommandsAddedFrom([
 $caller::class
]);
EOT;

        $this->lines->enqueue($message);

        return $this;
    }

    public function start(): void
    {
        $this->lines->enqueue('Will not start unsafe command.');
    }
}
