#!/usr/bin/perl
use List::Util qw(sum);

# Configuration
$noFolds = 5;

# Command-line parameters
$inputDir = $ARGV[0];
$mode = $ARGV[1];

# Read dataset
open (my $fh, '<', $inputDir . '/dataset-balanced.csv');
    my @lines = <$fh>;
    @lines = @lines[ 1 .. $#lines ]; #skip first line
close($fh);

# Cross-validation loop
for ($fold=0; $fold < $noFolds; $fold++) {
    # Open training, validation and test file
    open (my $test_fh,       '>test-data-file-' . $fold .'.csv');
    open (my $training_fh,   '>training-data-file-' . $fold . '.csv');
    open (my $validation_fh, '>validation-data-file-' . $fold . '.csv');

    # Separate training and test data according to fold
    $c=0;
    $tc=0;
    $vc=0;
    foreach $line (@lines) {
        if (($c - $fold) % 5 == 0) {
            # Test item, strip data to be predicted
            @fields = split(';', $line);
            $strip_line = $fields[0] . ';;;;;;;' . "\n";
            #$strip_line = $fields[1] . ';' . $fields[3] . ';' . $fields[4] . ';' . $fields[5] . ';' . $fields[6] . "\n";

            $tperformer[$tc]  = $fields[1];
            $tinstrument[$tc] = $fields[3];
            $tstyle[$tc]      = $fields[4];
            $tyear[$tc]       = $fields[5];
            $ttempo[$tc]      = $fields[6];

            print $test_fh $strip_line;
            $tc = $tc + 1;
        } elsif (($c - $fold) % 5 == 1) {
            @fields = split(';', $line);
            $strip_line = $fields[0] . ";;;;;;;" . "\n";

            $vperformer[$vc]  = $fields[1];
            $vinstrument[$vc] = $fields[3];
            $vstyle[$vc]      = $fields[4];
            $vyear[$vc]       = $fields[5];
            $vtempo[$vc]      = $fields[6];

            print $validation_fh $strip_line;
            $vc = $vc + 1;
        } else {
            # Training item
            print $training_fh $line;
        }

        $c=$c+1;
    }

    # Close files
    close($test_fh);
    close($training_fh);
    close($validation_fh);

    # Run classifier
    if ($mode eq '--validate') {
        system('./classification.py training-data-file-' . $fold .'.csv validation-data-file-' .$fold . '.csv output-file-' . $fold . '.csv');
    } elsif ($mode eq '--test') {
        system('./classification.py training-data-file-' . $fold .'.csv test-data-file-' .$fold . '.csv output-file-' . $fold . '.csv');
    }

    # Compare outputs with data that was stripped
    open (my $fh, '<', 'output-file-' . $fold .'.csv');
    my @lines = <$fh>;

    # Keep score
    $c=0;
    for $line(@lines) {
        @field = split(';', $line);

        if ($mode eq '--validate') {
            $performerPerformance[$fold] += !($field[0] eq $vperformer[$c]);
            $instrumentPerformance[$fold] += !($field[1] eq $vinstrument[$c]);
            $stylePerformance[$fold] += !($field[2] eq $vstyle[$c]);
            $yearPerformance[$fold] += abs($field[3] - $vyear[$c]);
            $tempoPerformance[$fold] += abs($field[4] - $vtempo[$c]);
        } elsif ($mode eq '--test') {
            $performerPerformance[$fold] += !($field[0] eq $tperformer[$c]);
            $instrumentPerformance[$fold] += !($field[1] eq $tinstrument[$c]);
            $stylePerformance[$fold] += !($field[2] eq $tstyle[$c]);
            $yearPerformance[$fold] += abs($field[3] - $tyear[$c]);
            $tempoPerformance[$fold] += abs($field[4] - $ttempo[$c]);
        }

        $c=$c+1;

    }
}

# Aggregate score
print "\n\n\n";
print "-----------------------------------\n";
if ($mode eq '--validate') {
    print "VALIDATION SET:\n";
} elsif ($mode eq '--test') {
    print "TEST SET:\n";
}
print "Error performance (lower is better)\n";
print "-----------------------------------\n";
print "Performer prediction\t" . join(';', @performerPerformance) . " => " . sum(@performerPerformance) ."\n";
print "Instrument prediction\t" . join(';', @instrumentPerformance) . " => " . sum(@instrumentPerformance) ."\n";
print "Style prediction\t" . join(';', @stylePerformance) . " => " . sum(@stylePerformance) ."\n";
print "Year prediction\t\t" . join(';', @yearPerformance) . " => " . sum(@yearPerformance) ."\n";
print "Tempo prediction\t" . join(';', @tempoPerformance) . " => " . sum(@tempoPerformance) ."\n";
