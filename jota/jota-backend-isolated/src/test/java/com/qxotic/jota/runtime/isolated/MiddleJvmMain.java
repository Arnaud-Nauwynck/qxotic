package com.qxotic.jota.runtime.isolated;

import java.nio.file.Path;

/**
 * The "parent" of the parent-death test, run in its own JVM: starts a child, prints its pid, then
 * waits to be killed. Killing this JVM must take the child down with it.
 */
public final class MiddleJvmMain {

    public static void main(String[] args) throws Exception {
        IsolatedJvm child =
                IsolatedJvm.start(
                        IsolatedJvmTest.testConfig().workDirectory(Path.of(args[0])).build());
        System.out.println("child-pid " + child.pid());
        System.out.flush();
        Thread.sleep(Long.MAX_VALUE);
    }
}
