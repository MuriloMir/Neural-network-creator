// This software is a visual interface for you to draw a neural network, then you just press Enter and it generates the source code of the network.

// import all the tools we need
import multimedia.audio : AudioOutputThread;
import multimedia.display : Color, Image, KeyEvent, Key, MouseButton, MouseEvent, MouseEventType, OperatingSystemFont, Point, Rectangle, ScreenPainter,
                            SimpleWindow;
import multimedia.image : loadImageFromMemory, memory;
import std.algorithm.iteration : map;
import std.algorithm.mutation : remove;
import std.algorithm.searching : countUntil, canFind;
import std.array : replace, split;
import std.conv : to;
import std.file : write;
import std.random : uniform;
import std.string : format;

// in case you are on Windows
version (Windows)
{
    // these 2 lines will simply stop the terminal from popping-up
    pragma(linkerDirective, "/subsystem:windows");
    pragma(linkerDirective, "/entry:mainCRTStartup");
}

// load the sounds which the GUI will play, they must be global because different functions will use them
memory neuronSelection = cast(memory) import("sounds/neuron selection.ogg"), neuronCreation = cast(memory) import("sounds/neuron creation.ogg"),
       neuronDeletion = cast(memory) import("sounds/neuron deletion.ogg"), synapseCreation = cast(memory) import("sounds/synapse creation.ogg"),
       synapsesDeletion = cast(memory) import("sounds/synapses deletion.ogg"), focus = cast(memory) import("sounds/focus.ogg"),
       clear = cast(memory) import("sounds/clear.ogg"), sourceCodeCreation = cast(memory) import("sounds/source code creation.ogg");

// this struct will represent the lines connecting the neurons, meaning the synapses
struct SynapseLine
{
    // it contains the name of the synapse and the rectangle of the next neuron in the synapse
    string name;
    Rectangle nextNeuronRect;

    // this is the constructor
    this (string name, Rectangle nextNeuronRect)
    {
        this.name = name;
        this.nextNeuronRect = nextNeuronRect;
    }
}

// this class will represent each neuron you create on the screen
class NeuronImage
{
    // it contains the rectangle where the neuron is, the name of the neuron and the color of the forward synapses of the neuron
    Rectangle rect;
    string name;
    Color synapseColor;
    // it also contains the backward and forward synapses
    SynapseLine[] backwardSynapses, forwardSynapses;

    // this is the constructor
    this (Rectangle rect, string name)
    {
        this.rect = rect;
        this.name = name;
        // create a random color, that way each neuron has a unique color
        this.synapseColor = Color(uniform(100, 256), uniform(100, 256), uniform(100, 256));
    }
}

// this function will draw a neuron on the screen and write its name inside it
void drawNeuron(NeuronImage neuron, Color neuronColor, OperatingSystemFont font, ScreenPainter painter)
{
    // set the color of the painter
    painter.outlineColor = neuron.synapseColor, painter.fillColor = neuron.synapseColor;
    // draw a halo with the color of the neuron's synapses, to make it easy to identify it
    painter.drawRectangle(neuron.rect.upperLeft(), 30, 30);
    // set the color of the painter
    painter.outlineColor = neuronColor, painter.fillColor = neuronColor;
    // draw the actual neuron
    painter.drawRectangle(Point(neuron.rect.upperLeft().x + 2, neuron.rect.upperLeft().y + 2), 26, 26);
    // set the color of the painter
    painter.outlineColor = Color.blue();
    // set the font to be used by the painter
    painter.setFont(font);
    // write the name of the neuron
    painter.drawText(Point(neuron.rect.upperLeft().x + 3, neuron.rect.upperLeft().y + 8), neuron.name);
}

// this function will delete a neuron from a layer, 'i' is ulong because it will be an index from a 'foreach' loop
void deleteNeuron(ref NeuronImage[] layer, ref int latestNeuron, int neuronNumber, ulong i)
{
    // if you are removing the neuron with the highest number, the number is in the name
    if (neuronNumber == latestNeuron)
        // decrement the counter, it always start counting from the latest one
        latestNeuron--;

    // remove the neuron from the layer
    layer = remove(layer, i);

    // if you've deleted all neurons from the layer
    if (layer.length == 0)
        // the "latest" variable has to be 0 again
        latestNeuron = 0;
}

// this function will generate the source code with the neural network
void generateCode(NeuronImage[][5] allLayers, AudioOutputThread sounds)
{
    // this mutable string will contain the text with the source code
    char[] finalText = cast(char[]) "// create the neurons\nNeuron ";
    // this string will be used when creating the statement with the layer definitions
    string layerDefinitionString;

    // start a loop to add the declaration of all neurons of all layers
    foreach (layer; allLayers)
    {
        // start a loop to check all neurons of a layer
        foreach (neuron; layer)
            // add the statement to the final text
            finalText ~= format("%s = new Neuron(\"%s\"), ", neuron.name, neuron.name);

        // if the layer has any neurons in it
        if (layer != [])
            // remove the last space in order to finish each layer with a new line, then add 7 spaces to keep the following layer indented
            finalText[$ - 1] = '\n', finalText ~= "       ";
    }

    // remove the 7 spaces and replace the ',' with a ';' in order to finish it all with a ';' and a new line
    finalText = finalText[0 .. $ - 7], finalText[$ - 2] = ';';
    // add the declaration of all synapses
    finalText ~= "\n// create the synapses\nSynapse ";

    // start a loop to add the declaration of all synapses
    foreach (layer; allLayers)
        // start a loop to check all neurons of a layer
        foreach (neuron; layer)
        {
            // start a loop to check all forward synapses of the neuron
            foreach (synapse; neuron.forwardSynapses)
                // add the statement, use the synapse's name and split it where the "to" word is found, in order to obtain the 1st and 2nd neurons
                finalText ~= format("%s = new Synapse(uniform(-2.0, 2.0), %s, %s), ", synapse.name, split(synapse.name, "to")[0],
                                                                                      split(synapse.name, "to")[1]);

            // if the neuron has forward synapses (the output neuron doesn't have them)
            if (neuron.forwardSynapses != [])
                // finish the declaration with a new line and add 8 spaces to keep the following layer indented
                finalText[$ - 1] = '\n', finalText ~= "        ";
        }

    // make the end contain only a ';' and a new line, therefore remove the 8 spaces and then replace the ',' with a ';'
    finalText = finalText[0 .. $ - 8], finalText[$ - 2] = ';', finalText ~= "\n// give the neurons their synapses\n";

    // start a loop to define the arrays of backward and forward synapses of all neurons
    foreach (layer; allLayers)
        // start a loop to check all neurons of a layer
        foreach (neuron; layer)
        {
            // if the neuron has backward synapses (the input neurons don't have them)
            if (neuron.backwardSynapses != [])
                // use 'map()' to simulate list comprehension and get a list with the names of all backward synapses, the "" are removed from the names
                finalText ~= replace(format("%s.backwardSynapses = %s,\n", neuron.name, map!(x => x.name)(neuron.backwardSynapses)), "\"", "");

            // if the neuron has forward synapses (the output neuron doesn't have them)
            if (neuron.forwardSynapses != [])
                // use 'map()' to simulate list comprehension and get a list with the names of all backward synapses, the "" are removed from the names
                finalText ~= replace(format("%s.forwardSynapses = %s,\n", neuron.name, map!(x => x.name)(neuron.forwardSynapses)), "\"", "");
        }

    // replace the last ',' with a ';'
    finalText[$ - 2] = ';';
    // create the text with the definition of the layers
    finalText ~= "\n// create the layers\n";
    // use 'map()' to simulate list comprehension and get a list with the names of all neurons in each layer, then format them into a statement
    layerDefinitionString = format("Neuron[] layerA = %s, layerB = %s, layerC = %s, layerD = %s, layerE = %s;\n",
                                   map!(x => x.name)(allLayers[0]), map!(x => x.name)(allLayers[1]), map!(x => x.name)(allLayers[2]),
                                   map!(x => x.name)(allLayers[3]), map!(x => x.name)(allLayers[4]));
    // remove the "" from the names and add the statement to the final text
    finalText ~= replace(layerDefinitionString, "\"", "");
    // add this statement to the final text
    finalText ~= "// this array will contain all layers\nNeuron[][] allLayers = [layerA, layerB, layerC, layerD, layerE];\n";

    // if the 3rd layer is emtpy, it's necessary to remove the definition of empty layers because it will cause bugs
    if (allLayers[2] == [])
        // remove the empty layer's statements from the final text
        finalText = replace(finalText, ", layerC = [], layerD = [], layerE = []", ""), finalText = replace(finalText, ", layerC, layerD, layerE", "");
    // if the 4th layer is emtpy
    else if (allLayers[3] == [])
        // remove the empty layer's statements from the final text
        finalText = replace(finalText, ", layerD = [], layerE = []", ""), finalText = replace(finalText, ", layerD, layerE", "");
    // if the 5th layer is emtpy
    else if (allLayers[4] == [])
        // remove the empty layer's statements from the final text
        finalText = replace(finalText, ", layerE = []", ""), finalText = replace(finalText, ", layerE", "");

    // finish it with the definition of the final variables
    finalText ~= "// create the network\nNetwork net = new Network(allLayers);\n\n// these variables will be used in the training process\nfloat actualResult, expectedResult, error;\n// set the learning rate\nlearningRate = 1.0;\n\n";

    // start a loop to write all synapses with their weights, so you can have all weights to save them in a text file later
    foreach (layer; allLayers)
        // start a loop to check all neurons of a layer
        foreach (neuron; layer)
            // start a loop to check all forward synapses
            foreach (synapse; neuron.forwardSynapses)
                // add the name and the weight to the final text, that way you will have a list with them all
                finalText ~= format("%s.weight, ", synapse.name);

    // remove the extra " ," in the end
    finalText = finalText[0 .. $ - 2];
    // create the source code file with the text
    write("source code.d", finalText);
    // play the source code creation sound
    sounds.playOgg(sourceCodeCreation);
}

// start the software
void main()
{
    // create arrays to represent layers, 'currentLayer', 'layerInFront' and 'layerBehind' are used when deleting a neuron's synapses or a neuron itself
    NeuronImage[] layerA, layerB, layerC, layerD, layerE, currentLayer, layerInFront, layerBehind;
    // create the GUI
    SimpleWindow window = new SimpleWindow(1250, 800, "Neural network creator");
    // these variables will be used when you are connecting 2 neurons
    NeuronImage firstSelectedNeuron, secondSelectedNeuron;
    // this variable will hold the rectangle of a new neuron being created
    Rectangle newRect;
    // these booleans will tell the event loop if you have selected a neuron or if you have focused on one
    bool selected, focusModeOn, dialogOn;
    // this variable will be used to know the name of the next layer when we delete a neuron's forward synapses
    char nextLayerName;
    // these ints will tell the number of the latest neuron of each layer, to be used when formatting the names of new neurons being created
    int latestA, latestB, latestC, latestD, latestE;
    // this string will contain the name of a new synapse which is being created
    string synapseName;
    // this array will contain the names of all synapses in the network
    string[] existingSynapses;
    // these will be the rectangles where the "Yes" and "No" buttons will be when the dialog box is on
    Rectangle yesButton = Rectangle(437, 437, 572, 501), noButton = Rectangle(675, 437, 810, 501);
    // these will the points where the instructions bar and the dialog box will be drawn
    Point instructionsBarPoint = Point(0, 750), dialogBoxPoint = Point(350, 220);
    // create the image of the instructions bar, which is displayed at the bottom, and the image of the dialog box, displayed when you press Escape
    Image instructionsBarImg = Image.fromMemoryImage(loadImageFromMemory(cast(memory) import("instructions.jpeg"))),
          dialogBoxImg = Image.fromMemoryImage(loadImageFromMemory(cast(memory) import("dialog.jpeg")));
    // this will be the audio thread, to play sounds
    AudioOutputThread sounds = AudioOutputThread(true);

    // in case you are on Windows
    version(Windows)
        // create the font to be used to draw the names of the neurons
        OperatingSystemFont font = new OperatingSystemFont("Noto Mono", 15);
    // in case you are on Linux
    else
        // create the font to be used to draw the names of the neurons
        OperatingSystemFont font = new OperatingSystemFont("Ubuntu", 10);

    // this nested function will register whenever you right-click on the screen to create a new neuron or when you left-click to select a neuron
    void neuronInteraction(MouseEvent event, ref NeuronImage[] layer, string layerName, ref int latestNeuron)
    {
        // if you just click to exit the focus mode
        if (focusModeOn)
        {
            // undo any selection you had done before
            firstSelectedNeuron = null;
            // set this boolean to false
            focusModeOn = false;

            // finish the function
            return;
        }

        // set this boolean to false in order to check if you've selected a neuron, 'selected' will become true if you've clicked on one
        selected = false;

        // start a loop to check all neurons in this layer to see if you have clicked on 1 of them
        foreach (ref neuron; layer)
            // if you've clicked on the neuron
            if (neuron.rect.contains(Point(event.x, event.y)))
            {
                // if you have not selected a neuron before, in which case this will be the 1st and not the 2nd selected neuron
                if (firstSelectedNeuron is null)
                    // make this neuron be the first selected one
                    firstSelectedNeuron = neuron;
                // if the layer's letter differece is only 1 (you can only connect a neuron to another one in the layer right front of it)
                else if (neuron.name[0] - firstSelectedNeuron.name[0] == 1)
                    // make this the second selected neuron
                    secondSelectedNeuron = neuron;
                // if you try to connect a neuron to another one behind or beside it
                else
                    // the only effect this has is to change the selection to the other neuron you've clicked
                    firstSelectedNeuron = neuron;

                // let the event loop know that some neuron has been selected
                selected = true;

                // finish the loop, we are done
                break;
            }

        // if you have right-clicked to add a new neuron or to focus on a specific one
        if (event.button == MouseButton.right)
        {
            // start a loop to check if you have right-clicked a neuron, which causes it to focus on it
            foreach (neuron; layer)
                // if you've right-clicked the first or the second selected neuron
                if (neuron is firstSelectedNeuron || neuron is secondSelectedNeuron)
                {
                    // store it in the 1st selected neuron and forget the 2nd selected neuron (we only need the 1st selected neuron for this)
                    firstSelectedNeuron = neuron, secondSelectedNeuron = null;
                    // let the event loop know the focus mode is now on
                    focusModeOn = true;
                    // play the focus sound
                    sounds.playOgg(focus);

                    // finish the loop, we are done
                    break;
                }

            // if you're not on focus mode, in this case you have right-clicked on empty space to create a new neuron
            if (!focusModeOn)
            {
                // if you've right-clicked at least 80 pixels above the instructions bar (50 pixels of the bar + 30 pixels of the neuron box)
                if (event.y < 720)
                    // if the neuron is being created on top of the division line between layers
                    if (layerName == "A" && event.x > 220 || layerName == "B" && event.x > 470 || layerName == "C" && event.x > 720 ||
                        layerName == "D" && event.x > 970 || layerName == "E" && event.x > 1220)
                        // create the rectangle of the neuron by shifting it to the side, so it doesn't touch the line
                        newRect = Rectangle(event.x - 30, event.y, event.x, event.y + 30);
                    // if it is being created in the right place
                    else
                        // create the rectangle of the neuron
                        newRect = Rectangle(event.x, event.y, event.x + 30, event.y + 30);
                // if you've right-clicked too close to the instructions bar
                else
                    // create the rectangle of the neuron after correcting the y coordinate
                    newRect = Rectangle(event.x, 720, event.x + 30, 750);

                // create a name for it, it names them according to the order they were created, using 'latest' variables (after incrementing it)
                string name = layerName ~ to!string(++latestNeuron);
                // add it to the corresponding layer
                layer ~= new NeuronImage(newRect, name);
                // play the neuron creation sound
                sounds.playOgg(neuronCreation);
            }
        }
        // if you have selected a second neuron, then we will connect the first with the second
        else if (selected && secondSelectedNeuron !is null)
        {
            // create the name of the synapse
            synapseName = firstSelectedNeuron.name ~ "to" ~ secondSelectedNeuron.name;

            // if this synapse doesn't exist, we have to make sure you are not creating duplicated synapses
            if (!canFind(existingSynapses, synapseName))
            {
                // give it to the second selected neuron as a backward synapse
                secondSelectedNeuron.backwardSynapses ~= SynapseLine(synapseName, secondSelectedNeuron.rect);
                // give it to the first selected neuron as a forward synapse
                firstSelectedNeuron.forwardSynapses ~= SynapseLine(synapseName, secondSelectedNeuron.rect);
                // it to the array of existing synapses
                existingSynapses ~= synapseName;
                // play the synapse creation sound
                sounds.playOgg(synapseCreation);
            }

            // set them back to 'null', so they can be reused later
            firstSelectedNeuron = secondSelectedNeuron = null;
        }
        // if you've just selected a neuron
        else if (selected)
            // play the neuron selection sound
            sounds.playOgg(neuronSelection);
        // if you've clicked on an empty area
        else
            // set them back to 'null' in order to remove all selections
            firstSelectedNeuron = secondSelectedNeuron = null;
    }

    // start the event loop of the GUI, with 50msecs of refresh time
    window.eventLoop(50,
    {
        // here we paint the layers and the neurons
        ScreenPainter painter = window.draw();
        // we start painting it all black
        painter.clear(Color.black());
        // then we draw the white lines
        painter.outlineColor = Color.white();
        // we draw lines dividing the screen in 4 layers
        painter.drawLine(Point(250, 0), Point(250, 800)), painter.drawLine(Point(500, 0), Point(500, 800)), painter.drawLine(Point(750, 0), Point(750, 800)),
            painter.drawLine(Point(1000, 0), Point(1000, 800));

        // start a loop to check all layers, in order to draw the neurons of all layers
        foreach (layer; [layerA, layerB, layerC, layerD, layerE])
            // start a loop to check all neurons of the layer
            foreach (neuron; layer)
                // if you have focused on a neuron then we only show that specific neuron and the neurons it is connected to
                if (focusModeOn)
                    // if it is the neuron you've focused on
                    if (neuron is firstSelectedNeuron)
                    {
                        // start a loop to draw all the synapses
                        foreach (synapse; neuron.forwardSynapses)
                        {
                            // get the color of the neuron
                            painter.outlineColor = neuron.synapseColor;
                            // draw the line connecting the neurons
                            painter.drawLine(neuron.rect.center(), synapse.nextNeuronRect.center());
                        }

                        // draw the neuron, notice we use the color yellow instead of gray, to make it obvious you are in the focus mode
                        drawNeuron(neuron, Color.yellow(), font, painter);
                    }
                    // if it is one of the neurons that are connected to the focused neuron
                    else
                    {
                        // start a loop to check all forward synapses, we first draw the neurons that are forward connected to the focused neuron
                        foreach (synapse; neuron.forwardSynapses)
                            // if one of its forward synapses contains the name of the focused neuron
                            if (synapse.name[countUntil(synapse.name, "to") + 2 .. $] == firstSelectedNeuron.name)
                            {
                                // get the color of the neuron
                                painter.outlineColor = neuron.synapseColor;
                                // draw the synapse
                                painter.drawLine(neuron.rect.center(), synapse.nextNeuronRect.center());
                                // draw the neuron
                                drawNeuron(neuron, Color.yellow(), font, painter);

                                // end the loop because we don't need to check any other synapse after this
                                break;
                            }

                        // start a loop to check all backward synapses, we draw the neurons that are backward connected to the focused neuron
                        foreach (synapse; neuron.backwardSynapses)
                            // if one of its backward synapses contains the name of the focused neuron
                            if (synapse.name[0 .. countUntil(synapse.name, "to")] == firstSelectedNeuron.name)
                            {
                                // draw the neuron, we don't need to draw the synapse here because that has already been done above
                                drawNeuron(neuron, Color.yellow(), font, painter);

                                // end the loop because we don't need to check any other synapse after this
                                break;
                            }
                    }
                // if you are not on focus mode, then just draw it all normally
                else
                {
                    // draw the synapses connecting the neurons, they must be drawn first, so the neuron image will be on top of them
                    painter.outlineColor = neuron.synapseColor;

                    // start a loop to check all forward synapses
                    foreach (synapse; neuron.forwardSynapses)
                        // connect them by their centers
                        painter.drawLine(neuron.rect.center(), synapse.nextNeuronRect.center());

                    // if it is the first selected neuron
                    if (neuron == firstSelectedNeuron)
                        // draw it as a red square, so it stands out
                        drawNeuron(neuron, Color.red(), font, painter);
                    // if it is any other unselected neuron
                    else
                        // draw it as a gray square
                        drawNeuron(neuron, Color.gray(), font, painter);
                }

        // draw the instructions bar at the bottom
        painter.drawImage(instructionsBarPoint, instructionsBarImg);

        // if the dialog box is on
        if (dialogOn)
            // draw the dialog box in the center
            painter.drawImage(dialogBoxPoint, dialogBoxImg);
    },
    // register mouse events
    (MouseEvent event)
    {
        // if you've pressed a mouse button, notice the y coordinate must be less than 750 because the instructions bar is 50 pixels tall
        if (event.type == MouseEventType.buttonPressed)
            // if the dialog box is on the screen and it was a left-click
            if (dialogOn)
            {
                // if you click on "Yes" and it was a left-click, then we erase everything
                if (yesButton.contains(Point(event.x, event.y)) && event.button == MouseButton.left)
                {
                    // delete all neurons, they are classes, therefore each needs to be deleted individually
                    layerA = [], layerB = [], layerC = [], layerD = [], layerE = [];
                    // delete all synapses;
                    existingSynapses = [];
                    // set the selected neuron back to null because now it no longer exists
                    firstSelectedNeuron = null;
                    // play the clear sound
                    sounds.playOgg(clear);
                    // set this boolean to false
                    dialogOn = false;
                }
                // if you click on "No" and it was a left-click
                else if (noButton.contains(Point(event.x, event.y)) && event.button == MouseButton.left)
                    // set this boolean to false
                    dialogOn = false;
            }
            else
            {
                // if you are in the 1st layer
                if (event.x < 250 && event.y < 750)
                    // interact with the neuron to create one, select one or add synapses to one, we use the 'neuronInteraction()' function above
                    neuronInteraction(event, layerA, "A", latestA);
                // if you are in the 2nd layer
                else if (250 < event.x && event.x < 500 && event.y < 750)
                    // interact with the neuron to create one, select one or add synapses to one, we use the 'neuronInteraction()' function above
                    neuronInteraction(event, layerB, "B", latestB);
                // if you are in the 3rd layer
                else if (500 < event.x && event.x < 750 && event.y < 750)
                    // interact with the neuron to create one, select one or add synapses to one, we use the 'neuronInteraction()' function above
                    neuronInteraction(event, layerC, "C", latestC);
                // if you are in the 4th layer
                else if (750 < event.x && event.x < 1000 && event.y < 750)
                    // interact with the neuron to create one, select one or add synapses to one, we use the 'neuronInteraction()' function above
                    neuronInteraction(event, layerD, "D", latestD);
                // if you are in the 5th layer
                else if (1000 < event.x && event.x < 1250 && event.y < 750)
                    // interact with the neuron to create one, select one or add synapses to one, we use the 'neuronInteraction()' function above
                    neuronInteraction(event, layerE, "E", latestE);
            }
    },
    // register keyboard events
    (KeyEvent event)
    {
        // if you've pressed something and the dialog box is not on the screen
        if (event.pressed && !dialogOn)
            // if you've pressed Enter
            if (event.key == Key.Enter)
                // generate the neural network's source code
                generateCode([layerA, layerB, layerC, layerD, layerE], sounds);
            // if you've pressed Escape
            else if (event.key == Key.Escape)
                // set this boolean to true
                dialogOn = true;
            // if a neuron is selected
            else if (firstSelectedNeuron !is null)
                // if you've pressed Control, in this case we delete all forward synapses of the neuron
                if (event.key == Key.Ctrl)
                {
                    // find the next layer's letter through the name, so we can remove it from the list of backward synapses of the next neurons
                    nextLayerName = cast(char) (firstSelectedNeuron.name[0] + 1);

                    // if the next layer is the 2nd
                    if (nextLayerName == 'B')
                        // assign it to 'layerInFront'
                        layerInFront = layerB;
                    // if the next layer is the 3rd
                    else if (nextLayerName == 'C')
                        // assign it to 'layerInFront'
                        layerInFront = layerC;
                    // if the next layer is the 4th
                    else if (nextLayerName == 'D')
                        // assign it to 'layerInFront'
                        layerInFront = layerD;
                    // if the next layer is the 5th
                    else if (nextLayerName == 'E')
                        // assign it to 'layerInFront'
                        layerInFront = layerE;

                    // start a loop to check all synapses from the selected neuron
                    foreach (synapse; firstSelectedNeuron.forwardSynapses)
                    {
                        // remove the synapse from the list of existing synapses
                        existingSynapses = remove(existingSynapses, countUntil(existingSynapses, synapse.name));

                        // start a loop to check all neurons in the next layer
                        foreach (ref neuron; layerInFront)
                            // if this synapse belongs to the neuron we are analyzing right now
                            if (canFind(neuron.backwardSynapses, synapse)) 
                                // remove it from the list of backward synapses
                                neuron.backwardSynapses = remove(neuron.backwardSynapses, countUntil(neuron.backwardSynapses, synapse));
                    }

                    // if there are any synapses to be deleted
                    if (firstSelectedNeuron.forwardSynapses != [])
                    {
                        // clear the array of forward synapses of the selected neuron
                        firstSelectedNeuron.forwardSynapses = [];
                        // play the synapses deletion sound
                        sounds.playOgg(synapsesDeletion);
                    }
                }
                // if you've pressed Delete, in this case you can delete the neuron itself
                else if (event.key == Key.Delete)
                {
                    // check in which layer we are, by analyzing the selected neuron's name
                    switch (firstSelectedNeuron.name[0])
                    {
                        // if the selected neuron's layer is the 1st, get the current and the next layer
                        case 'A': currentLayer = layerA, layerInFront = layerB; break;
                        // if the selected neuron's layer is the 2nd, get the current and the next layer
                        case 'B': currentLayer = layerB, layerBehind = layerA, layerInFront = layerC; break;
                        // if the selected neuron's layer is the 3rd, get the current and the next layer
                        case 'C': currentLayer = layerC, layerBehind = layerB, layerInFront = layerD; break;
                        // if the selected neuron's layer is the 4th, get the current and the next layer
                        case 'D': currentLayer = layerD, layerBehind = layerC, layerInFront = layerE; break;
                        // if the selected neuron's layer is the 5th, get the current and the next layer
                        case 'E': currentLayer = layerE, layerBehind = layerD; break;
                        // in case something went wrong
                        default: assert(0);
                    }

                    // start a loop to check all forward synapses of the selected neuron
                    foreach (synapse; firstSelectedNeuron.forwardSynapses)
                    {
                        // remove the synapse from the list of existing synapses
                        existingSynapses = remove(existingSynapses, countUntil(existingSynapses, synapse.name));

                        // start a loop to check all neurons in the next layer
                        foreach (ref neuron; layerInFront)
                            // if this synapse belongs to the neuron we are analyzing right now
                            if (canFind(neuron.backwardSynapses, synapse)) 
                                // remove it from the list of backward synapses
                                neuron.backwardSynapses = remove(neuron.backwardSynapses, countUntil(neuron.backwardSynapses, synapse));
                    }

                    // start a loop to check all backward synapses of the selected neuron and remove them from the list of existing synapses
                    foreach (synapse; firstSelectedNeuron.backwardSynapses)
                    {
                        // remove the synapse from the list of existing synapses
                        existingSynapses = remove(existingSynapses, countUntil(existingSynapses, synapse.name));

                        // start a loop to check all neurons in the previous layer
                        foreach (ref neuron; layerBehind)
                            // if this synapse belongs to the neuron we are analyzing right now
                            if (canFind(neuron.forwardSynapses, synapse)) 
                                // remove it from the list of forward synapses
                                neuron.forwardSynapses = remove(neuron.forwardSynapses, countUntil(neuron.forwardSynapses, synapse));
                    }

                    // start a loop to check all neurons in the layer
                    foreach (i, neuron; currentLayer)
                        // if it is the selected neuron, we have to do it this way because they are classes, meaning the arrays only contain MAs
                        if (neuron == firstSelectedNeuron)
                        {
                            // check in which layer we are, we remove neurons this way because the layers are arrays containing the MAs of the classes
                            switch (neuron.name[0])
                            {
                                // if we are in layer A
                                case 'A': deleteNeuron(layerA, latestA, to!int(firstSelectedNeuron.name[1 .. $]), i); break;
                                // if we are in layer B
                                case 'B': deleteNeuron(layerB, latestB, to!int(firstSelectedNeuron.name[1 .. $]), i); break;
                                // if we are in layer C
                                case 'C': deleteNeuron(layerC, latestC, to!int(firstSelectedNeuron.name[1 .. $]), i); break;
                                // if we are in layer D
                                case 'D': deleteNeuron(layerD, latestD, to!int(firstSelectedNeuron.name[1 .. $]), i); break;
                                // if we are in layer E
                                case 'E': deleteNeuron(layerE, latestE, to!int(firstSelectedNeuron.name[1 .. $]), i); break;
                                // if something went wrong
                                default: assert(0);
                            }

                            // end the loop, we are done
                            break;
                        }

                    // set the selected neuron back to null because now it no longer exists
                    firstSelectedNeuron = null;
                    // play the neuron deletion sound
                    sounds.playOgg(neuronDeletion);
                }
    });
}
