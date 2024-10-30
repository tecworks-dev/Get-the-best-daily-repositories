package com.example.test.app;

import android.app.Activity;
import android.os.Build;
import android.os.Bundle;
import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

import dev.tmpfs.libcoresyscall.core.IAllocatedMemory;
import dev.tmpfs.libcoresyscall.core.MemoryAccess;
import dev.tmpfs.libcoresyscall.core.MemoryAllocator;
import dev.tmpfs.libcoresyscall.core.NativeHelper;
import dev.tmpfs.libcoresyscall.core.Syscall;

public class TestMainActivity extends Activity {

    private TextView mTestTextView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mTestTextView = new TextView(this);
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT);
        LinearLayout linearLayout = new LinearLayout(this);
        linearLayout.setLayoutParams(layoutParams);
        linearLayout.addView(mTestTextView);
        float dp8 = getResources().getDisplayMetrics().density * 8;
        linearLayout.setPadding((int) dp8, (int) dp8, (int) dp8, (int) dp8);
        setContentView(linearLayout);
        mTestTextView.setText(runTests());
    }

    private String runTests() {
        StringBuilder sb = new StringBuilder();
        sb.append("ISA = ").append(NativeHelper.getIsaName(NativeHelper.getCurrentRuntimeIsa()));
        sb.append("\n");
        sb.append("SDK_INT = ").append(Build.VERSION.SDK_INT);
        sb.append("\n");
        sb.append("Page size = ").append(Os.sysconf(OsConstants._SC_PAGESIZE));
        sb.append("\n");
        int __NR_uname;
        switch (NativeHelper.getCurrentRuntimeIsa()) {
            case NativeHelper.ISA_X86_64:
                __NR_uname = 63;
                break;
            case NativeHelper.ISA_ARM64:
                __NR_uname = 160;
                break;
            case NativeHelper.ISA_X86:
            case NativeHelper.ISA_ARM:
                __NR_uname = 122;
                break;
            default:
                // just for demo purpose. I don't want to search for the correct value for other ISAs.
                throw new IllegalStateException("Unexpected value: " + NativeHelper.getCurrentRuntimeIsa());
        }
        ///** The maximum length of any field in `struct utsname`. */
        //#define SYS_NMLN 65
        ///** The information returned by uname(). */
        //struct utsname {
        //  /** The OS name. "Linux" on Android. */
        //  char sysname[SYS_NMLN];
        //  /** The name on the network. Typically "localhost" on Android. */
        //  char nodename[SYS_NMLN];
        //  /** The OS release. Typically something like "4.4.115-g442ad7fba0d" on Android. */
        //  char release[SYS_NMLN];
        //  /** The OS version. Typically something like "#1 SMP PREEMPT" on Android. */
        //  char version[SYS_NMLN];
        //  /** The hardware architecture. Typically "aarch64" on Android. */
        //  char machine[SYS_NMLN];
        //  /** The domain name set by setdomainname(). Typically "localdomain" on Android. */
        //  char domainname[SYS_NMLN];
        //};
        int sysnameOffset = 0;
        int nodenameOffset = 65;
        int releaseOffset = 65 * 2;
        int versionOffset = 65 * 3;
        int machineOffset = 65 * 4;
        int domainnameOffset = 65 * 5;
        int utsSize = 65 * 6;
        try (IAllocatedMemory uts = MemoryAllocator.allocate(utsSize, true)) {
            long utsAddress = uts.getAddress();
            Syscall.syscall(__NR_uname, utsAddress);
            sb.append("sysname = ").append(MemoryAccess.peekCString(utsAddress + sysnameOffset));
            sb.append("\n");
            sb.append("nodename = ").append(MemoryAccess.peekCString(utsAddress + nodenameOffset));
            sb.append("\n");
            sb.append("release = ").append(MemoryAccess.peekCString(utsAddress + releaseOffset));
            sb.append("\n");
            sb.append("version = ").append(MemoryAccess.peekCString(utsAddress + versionOffset));
            sb.append("\n");
            sb.append("machine = ").append(MemoryAccess.peekCString(utsAddress + machineOffset));
            sb.append("\n");
            sb.append("domainname = ").append(MemoryAccess.peekCString(utsAddress + domainnameOffset));
            sb.append("\n");
        } catch (ErrnoException e) {
            sb.append("ErrnoException: ").append(e.getMessage());
        }
        return sb.toString();
    }

}
